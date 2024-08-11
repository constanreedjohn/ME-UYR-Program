#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with CommonVoice.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on.

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper.yaml

Authors
 * Pooneh Mousavi 2022
 * Adel Moumen 2024
"""

import logging
import sys
import os

import torch
import torchaudio
import time
import faiss
import numpy as np
from tqdm import tqdm
from enum import Enum, auto
from dataclasses import dataclass
from hyperpyyaml import load_hyperpyyaml

from torch import nn
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.dataio.dataloader import LoopedLoader, SaveableDataLoader
from speechbrain.utils.distributed import if_main_process, run_on_main


logger = logging.getLogger(__name__)

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    SOURCE_TEST = auto()
    TARGET_TEST = auto()
    
@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision (AMP).

    Arguments
    ---------
    dtype : torch.dtype
        The dtype to use for AMP.
    """

    dtype: torch.dtype

    @classmethod
    def from_name(self, name):
        """Create an AMPConfig from a string name.

        Arguments
        ---------
        name : str
            The name of the AMPConfig to create.  Must be one of `fp32`,
            `fp16`, or `bf16`.

        Returns
        -------
        AMPConfig
            The AMPConfig corresponding to the name.
        """
        if name is None or name == "fp32":
            return AMPConfig(torch.float32)
        elif name == "fp16":
            return AMPConfig(torch.float16)
        elif name == "bf16":
            return AMPConfig(torch.bfloat16)
        else:
            raise ValueError(
                f"Specified autocast mode ({name}) incorrect, expected one of `fp32`, `fp16`, `bf16`."
            )

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, source_batch, target_batch, faiss_index, faiss_k, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        # Process source batch as supervised learning
        source_batch = source_batch.to(self.device)
        source_wavs, source_wav_lens = source_batch.sig
        source_bos_tokens, source_bos_tokens_lens = source_batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            source_wavs, source_wav_lens = self.hparams.wav_augment(source_wavs, source_wav_lens)
            source_bos_tokens = self.hparams.wav_augment.replicate_labels(source_bos_tokens)
            source_bos_tokens_lens = self.hparams.wav_augment.replicate_labels(
                source_bos_tokens_lens
            )

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        source_abs_tokens_lens = (source_bos_tokens_lens * source_bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(source_abs_tokens_lens.max(), device=self.device)[None, :]
            < source_abs_tokens_lens[:, None]
        )
        source_bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        source_enc_out, source_logits, _ = self.modules.whisper(source_wavs, source_bos_tokens)
        source_log_probs = self.hparams.log_softmax(source_logits)
        # print(f"[COMPUTE_FORWARD] SOURCE LOGITS: {source_logits} - {type(source_logits)} - {source_logits.shape}")
        
        if stage == Stage.TRAIN:
            # Get target logits
            self.modules.eval()
            with torch.no_grad():
                target_batch = target_batch.to(self.device)
                target_wavs, target_wav_lens = target_batch.sig
                target_bos_tokens, target_bos_tokens_lens = target_batch.tokens_bos
                target_abs_tokens_lens = (target_bos_tokens_lens * target_bos_tokens.shape[1]).long()
                pad_mask = (
                    torch.arange(target_abs_tokens_lens.max(), device=self.device)[None, :]
                    < target_abs_tokens_lens[:, None]
                )
                target_bos_tokens[~pad_mask] = self.tokenizer.pad_token_id
                target_enc_out, target_logits, _ = self.modules.whisper(target_wavs, target_bos_tokens)
                target_log_probs = self.hparams.log_softmax(target_logits)      # NLL of the inference target batch
            
            self.modules.train()
            source_embedding = source_enc_out.detach().cpu().numpy()    # (batchsize, 1500, 384)
            source_embedding = source_embedding.reshape((source_embedding.shape[0], -1)) # (batchsize, 576000)
            
            # print(f"[COMPUTE_FORWARD] SOURCE EMBEDDING: {source_embedding} - {source_embedding.shape}")
            faiss_distance, faiss_sample = faiss_index.search(source_embedding, faiss_k)
            
            # print(f"[COMPUTER_FORWARD] FAISS SAMPLE ID: {faiss_sample} - {faiss_sample.shape}")
            simliarity = [faiss_index.reconstruct_batch(i.shape[0]) for i in faiss_sample]
            simliarity = np.asarray(simliarity)
            simliarity = simliarity.reshape((simliarity.shape[0], 1500, 384)) # (batchsize, 576000) -> (batchsize, 1500, 384)
            
            # print(f"[COMPUTER_FORWARD] FAISS RECONSTRUCT: {simliarity} - {simliarity.shape}")
            target_filtered = torch.tensor(simliarity, dtype=torch.float32)
            target_filtered = target_filtered.to(self.device)
            retrieved_target = target_filtered
            
            # print(f"[COMPUTER_FORWARD] COMBINATION BETWEEN SOURCE AND TARGET WITH BATCH OF {source_enc_out.shape[0]}: {target_log_probs.shape}")
            # target_log_probs = torch.nn.functional.relu(target_logits)
            if retrieved_target.shape != source_enc_out.shape != target_enc_out.shape:
                logger.info(f"[COMPUTE_OBJECTS] DIFF DIMENSION WHERE COMBINE: {retrieved_target.shape} - SOURCE: {source_logits.shape} - TARGET: {target_logits.shape}")
                return [source_log_probs, None, source_wav_lens], [torch.rand([source_enc_out.shape[0], 1500, 384]), torch.rand([source_enc_out.shape[0], 1500, 384]), torch.rand([source_enc_out.shape[0], 1500, 384])]
            
            return [source_log_probs, None, source_wav_lens], [retrieved_target, source_enc_out, target_enc_out, target_log_probs]
        
        hyps = None
        if stage == Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                source_enc_out.detach(), source_wav_lens
            )
        elif stage in [Stage.SOURCE_TEST, Stage.TARGET_TEST]:
            hyps, _, _, _ = self.hparams.test_search(source_enc_out.detach(), source_wav_lens)

        return [source_log_probs, hyps, source_wav_lens], []

    def compute_objectives(self, source_predictions, target_prediction, source_batch, target_batch, stage):
        triplet_loss = nn.TripletMarginLoss()
        """Computes the loss NLL given predictions and targets."""
        
        (source_log_probs, hyps, source_wav_lens) = source_predictions
        source_batch = source_batch.to(self.device)
        ids = source_batch.id
        source_tokens_eos, source_tokens_eos_lens = source_batch.tokens_eos
        
        # Augment Labels
        if stage == Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            source_tokens_eos = self.hparams.wav_augment.replicate_labels(source_tokens_eos)
            source_tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                source_tokens_eos_lens
            )
        source_loss = self.hparams.nll_loss(
            source_log_probs, source_tokens_eos, length=source_tokens_eos_lens
        )
        target_loss = 0.0
        if stage == Stage.TRAIN:
            (retrieved_logits, source_logits, target_logits, target_log_probs) = target_prediction
            target_batch = target_batch.to(self.device)
            target_loss = triplet_loss(source_logits, retrieved_logits, target_logits)
            target_tokens_eos, target_tokens_eos_lens = target_batch.tokens_eos
            # target nll loss for both batch
            target_inference_loss = self.hparams.nll_loss(
                target_log_probs, target_tokens_eos, length=target_tokens_eos_lens
            )
            
        
        if stage != Stage.TRAIN:
            target_inference_loss = 0
            tokens, tokens_lens = source_batch.tokens

            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in hyps
            ]

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            if hasattr(self.hparams, "normalized_transcripts"):
                predicted_words = [
                    self.tokenizer.normalize(text).split(" ")
                    for text in predicted_words
                ]

                target_words = [
                    self.tokenizer.normalize(text).split(" ")
                    for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]
                target_words = [text.split(" ") for text in target_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            # print(f"[COMPUTE OBJECTIVES] LABEL: {target_words}")
            # print(f"[COMPUTE OBJECTIVES] PREDICTED: {predicted_words}")

        return source_loss, target_loss, target_inference_loss
    
    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        """Creates DataLoaders for Datasets.

        This is used by ``fit()`` and ``evaluate()`` if they just receive
        Datasets.

        Alternatively, this can be called from outside the Brain subclass.
        In that case, the DataLoader should be passed to ``fit()`` in place
        of the dataset.

        The Stage.TRAIN DataLoader is handled specially. It has extra args for
        shuffle and drop_last. In DDP a DistributedSampler is created (unless
        the dataset is an IterableDataset).

        NOTE
        ----
        Some important DataLoader arguments are passed via **loader_kwargs,
        e.g., batch_size, num_workers, pin_memory.

        NOTE
        ----
        By default, ``evaluate()`` specifies ckpt_prefix=None to stop the test
        DataLoader being added to the checkpointer. If you need to add a
        recoverable after saving checkpoints (e.g., at test time, after
        checkpointing the training), and still be able to recover reasonably,
        you should probably specify ``allow_partial_load=True``.

        Arguments
        ---------
        dataset : Dataset
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default collate_fn,
            unless specified in loader_kwargs.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        ckpt_prefix : str, None
            Prefix to use for SaveableDataLoader Checkpoint name. The Stage
            name is added to this to create the full key. Set to None to not
            save the DataLoader.
        **loader_kwargs : dict
            Additional keyword arguments to the DataLoader.
            E.g., batch_size, num_workers, pin_memory.

        Returns
        -------
        DataLoader for the input dataset
        """
        # TRAIN stage is handled specially.
        if stage == Stage.TRAIN:
            loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)
        # This commented-out code block is useful when one can ensure
        # metric reporting is DDP-valid for VALID & EVAL datasets.
        # elif self.distributed_launch:
        #     loader_kwargs = sb.dataio.dataloader.distributed_loader_specifics(
        #         self.distributed_launch, self.rank, dataset, loader_kwargs
        #     )
        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, **loader_kwargs
        )

        if (
            self.checkpointer is not None
            and ckpt_prefix is not None
            and (
                isinstance(dataloader, SaveableDataLoader)
                or isinstance(dataloader, LoopedLoader)
            )
        ):
            ckpt_key = ckpt_prefix + stage.name
            self.checkpointer.add_recoverable(ckpt_key, dataloader)
        return dataloader
    
    def load_faiss_index(self, faiss_index_path, target_train_set, enable):
        language = faiss_index_path.split("/")[-1]
        # if faiss_index_path is not None:
        #     files = sorted(os.listdir(faiss_index_path))
        #     if "faiss_index_last.index" in files:
        #         logger.info("[FIT] LOADING LAST INDEX")
        #         faiss_index = faiss.read_index(os.path.join(faiss_index_path, "faiss_index_last.index"))
        #         return faiss_index
        #     else:
        #         logger.info(f"[FIT] LOADING LATEST INDEX {files[-1]}")
        #         faiss_index = faiss.read_index(os.path.join(faiss_index_path, files[-1]))
        #         return faiss_index
        
        if os.path.exists(os.path.join(hparams['faiss_index_path'])):
            files = sorted(os.listdir(os.path.join(hparams['faiss_index_path'])))
            if "faiss_index_last.index" in files:
                logger.info("[FIT] LOADING LAST INDEX")
                faiss_index = faiss.read_index(os.path.join(hparams['faiss_index_path'], "faiss_index_last.index"))
                return faiss_index
            else:
                logger.info(f"[FIT] LOADING LATEST INDEX {files[-1]}")
                faiss_index = faiss.read_index(os.path.join(hparams['faiss_index_path'], files[-1]))
                return faiss_index
        elif not os.path.exists(os.path.join(hparams['faiss_index_path'])):
            logger.info(f"[FIT] NO SAVED INDEX FOUND - CREATING")
            faiss_index = self.build_faiss_index(target_train_set=target_train_set, enable=enable)
            return faiss_index
    
    def fit(
        self,
        epoch_counter,
        source_train_set,
        target_train_set,
        source_valid_set=None,
        target_valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={}
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        source_valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if source_valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.

        Returns
        -------
        None
        """
        if self.test_only:
            logger.info(
                "Test only mode, skipping training and validation stages."
            )
            return

        if not (
            isinstance(source_train_set, DataLoader)
            or isinstance(source_train_set, LoopedLoader)
        ):
            source_train_set = self.make_dataloader(
                source_train_set, stage=Stage.TRAIN, **train_loader_kwargs
            )
            
        if not (
            isinstance(target_train_set, DataLoader)
            or isinstance(target_train_set, LoopedLoader)
        ):
            target_train_set = self.make_dataloader(
                target_train_set, stage=Stage.TRAIN, **train_loader_kwargs
            )
            
        if source_valid_set is not None and not (
            isinstance(source_valid_set, DataLoader)
            or isinstance(source_valid_set, LoopedLoader)
        ):
            source_valid_set = self.make_dataloader(
                source_valid_set,
                stage=Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )
            
        if target_valid_set is not None and not (
            isinstance(target_valid_set, DataLoader)
            or isinstance(target_valid_set, LoopedLoader)
        ):
            target_valid_set = self.make_dataloader(
                target_valid_set,
                stage=Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()
        
        faiss_index = self.load_faiss_index(hparams['faiss_index_path'], target_train_set, enable)
            
        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(
                source_train_set=source_train_set, 
                target_train_set=target_train_set,
                epoch=epoch, 
                faiss_index=faiss_index,
                enable=enable
            )
            self._fit_valid(valid_set=source_valid_set, epoch=epoch, enable=enable)
            
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    target_valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.step = 0
                
                # Compute/store important stats
                stage_stats = {"loss": avg_valid_loss}
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                with open(hparams['train_log'], 'a') as fout:
                    fout.write(f"Target_valid_loss: {stage_stats['loss']} - target_valid_CER: {stage_stats['CER']} - target_valid_WER: {stage_stats['WER']}\n")
                logger.info(f"Target_valid_loss: {stage_stats['loss']} - target_valid_CER: {stage_stats['CER']} - target_valid_WER: {stage_stats['WER']}")
            

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    def build_faiss_index(self, target_train_set, enable):
        print("[BUILD FAISS INDEX] INGESTING TARGET ENCODER INTO INDEX...")
        faiss_index = faiss.IndexFlatL2(576000)
        os.makedirs(os.path.join(hparams['faiss_index_path']), exist_ok=True)
        with tqdm(
            target_train_set,
            total=len(target_train_set),
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            self.modules.eval()
            with torch.no_grad():
                for idx, target_batch in enumerate(t):
                    if idx % hparams['save_index_step'] == 0:
                        logger.info(f"[BUILD FAISS INDEX] SAVING INDEX AT BATCH SIZE INDEX {idx}")
                        faiss.write_index(faiss_index, os.path.join(hparams['faiss_index_path'], f"faiss_index_{idx}.index"))
                        
                    target_batch = target_batch.to(self.device)
                    target_wavs, target_wav_lens = target_batch.sig
                    target_bos_tokens, target_bos_tokens_lens = target_batch.tokens_bos

                    # We compute the padding mask and replace the values with the pad_token_id
                    # that the Whisper decoder expect to see.
                    target_abs_tokens_lens = (target_bos_tokens_lens * target_bos_tokens.shape[1]).long()
                    pad_mask = (
                        torch.arange(target_abs_tokens_lens.max(), device=self.device)[None, :]
                        < target_abs_tokens_lens[:, None]
                    )
                    target_bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

                    # Forward encoder + decoder
                    target_enc_out, target_logits, _ = self.modules.whisper(target_wavs, target_bos_tokens)
                    target_embedding = target_enc_out.detach().cpu().numpy()                        # (batchsize, 1500, 384)
                    target_embedding = target_embedding.reshape((target_embedding.shape[0], -1))    # (batchsize, 576000)
                    faiss_index.add(target_embedding)
                
                logger.info(f"[BUILD FAISS INDEX] SAVING INDEX AT LAST")
                faiss.write_index(faiss_index, os.path.join(hparams['faiss_index_path'], f"faiss_index_last.index"))
        print(f"[BUILD FAISS INDEX] FAISS TOTAL: {faiss_index.ntotal}")
            
        return faiss_index
    
    def _fit_train(self, source_train_set, target_train_set, epoch, enable, faiss_index):
        # Training stage
        self.on_stage_start(Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        steps_since_ckpt = 0
        with tqdm(
            zip(source_train_set, target_train_set),
            total=len(source_train_set) + len(target_train_set),
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            if self.profiler is not None:
                self.profiler.start()
            for source_batch, target_batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                steps_since_ckpt += 1
                loss = self.fit_batch(source_batch=source_batch, target_batch=target_batch, faiss_index=faiss_index)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                if self.profiler is not None:
                    self.profiler.step()
                    if self.profiler.step_num > self.tot_prof_steps:
                        logger.info(
                            "The profiler finished, training is stopped."
                        )
                        self.profiler.stop()
                        quit()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if self._should_save_intra_epoch_ckpt(
                    last_ckpt_time, steps_since_ckpt
                ):
                    # Checkpointer class will handle running this on main only
                    self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()
                    steps_since_ckpt = 0

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0

    def fit_batch(self, source_batch, target_batch, faiss_index):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    source_outputs, target_outputs = self.compute_forward(source_batch=source_batch, target_batch=target_batch, faiss_index=faiss_index, faiss_k=1, stage=Stage.TRAIN)
                    source_loss, target_loss, target_inference_loss = self.compute_objectives(
                        source_predictions=source_outputs, 
                        source_batch=source_batch, 
                        target_prediction=target_outputs,
                        target_batch=target_batch,
                        stage=Stage.TRAIN
                    )
            else:
                source_outputs, target_outputs = self.compute_forward(source_batch=source_batch, target_batch=target_batch, faiss_index=faiss_index, faiss_k=1, stage=Stage.TRAIN)
                source_loss, target_loss, target_inference_loss = self.compute_objectives(
                    source_predictions=source_outputs, 
                    source_batch=source_batch, 
                    target_prediction=target_outputs,
                    target_batch=target_batch,
                    stage=Stage.TRAIN
                )

            scaled_source_loss = self.scaler.scale(
                source_loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_source_loss)
            
            scaled_target_loss = self.scaler.scale(
                target_loss.mean() / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_target_loss)
            
            scaled_target_inference_loss = self.scaler.scale(
                target_inference_loss.mean() / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_target_inference_loss)
            
            logger.info(f"[FIT_BATCH] SOURCE_LOSS: {scaled_source_loss} - TARGET_LOSS: {scaled_target_loss} - TARGET_INFER_LOSS: {scaled_target_inference_loss}\n")
            loss = scaled_source_loss + scaled_target_loss + scaled_target_inference_loss
            loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(source_batch, source_outputs, source_loss, should_step)
        return loss.detach().cpu()
    
    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.step = 0
                self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)
                
    def evaluate(
        self,
        test_set,
        stage,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, stage, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(stage, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=stage)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(stage, avg_test_loss, None)
        self.step = 0
        return avg_test_loss
    
    @torch.no_grad()
    def evaluate_batch(self, source_batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.eval_precision)
        if self.use_amp:
            with torch.autocast(
                dtype=amp.dtype, device_type=torch.device(self.device).type
            ):
                source_outputs, target_outputs = self.compute_forward(source_batch=source_batch, target_batch=None, faiss_index=None, faiss_k=1, stage=stage)
                source_loss, target_loss, target_inference_loss = self.compute_objectives(
                    source_predictions=source_outputs, 
                    source_batch=source_batch, 
                    target_prediction=target_outputs,
                    target_batch=None,
                    stage=stage
                )
        else:
            source_outputs, target_outputs = self.compute_forward(source_batch=source_batch, target_batch=None, faiss_index=None, faiss_k=1, stage=stage)
            source_loss, target_loss, target_inference_loss = self.compute_objectives(
                source_predictions=source_outputs, 
                source_batch=source_batch, 
                target_prediction=target_outputs,
                target_batch=None,
                stage=stage
            )
        return source_loss.detach().cpu()
    
    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        print(stage)
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == Stage.VALID:
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == Stage.SOURCE_TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.source_test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
                    
        elif stage == Stage.TARGET_TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.target_test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
            


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    source_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["source_train_csv"],
        # replacements={"data_root": data_folder},
    )
    target_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["target_train_csv"],
        # replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        source_train_data = source_train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        target_train_data = target_train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        source_train_data = source_train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        target_train_data = target_train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        source_train_data = source_train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # source_train_data = source_train_data.batch_shuffle(hparams["train_loader_kwargs"]["batch_size"])
        target_train_data = target_train_data.batch_shuffle(hparams["train_loader_kwargs"]["batch_size"])

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    source_valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["source_valid_csv"],
        # replacements={"data_root": data_folder},
    )
    source_valid_data = source_valid_data.filtered_sorted(sort_key="duration")
    
    target_valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["target_valid_csv"],
        # replacements={"data_root": data_folder},
    )
    target_valid_data = target_valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    target_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["target_test_csv"],
        # replacements={"data_root": data_folder},
    )
    
    source_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["source_test_csv"],
        # replacements={"data_root": data_folder},
    )

    datasets = [source_train_data, target_train_data, source_valid_data, target_valid_data, target_test_data, source_test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.sample_rate != hparams["sample_rate"]:
            sig = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"]
            )(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        if hasattr(hparams, "normalized_transcripts"):
            wrd = tokenizer.normalize(wrd)
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )

    return source_train_data, target_train_data, source_valid_data, target_valid_data, target_test_data, source_test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    # from common_voice_prepare import prepare_common_voice  # noqa

    # multi-gpu (ddp) save data preparation
    # run_on_main(
    #     prepare_common_voice,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_folder": hparams["save_folder"],
    #         "source_train_tsv_file": hparams["source_train_tsv_file"],
    #         "target_train_tsv_file": hparams["target_train_tsv_file"],
    #         "dev_tsv_file": hparams["dev_tsv_file"],
    #         "test_tsv_file": hparams["test_tsv_file"],
    #         "accented_letters": hparams["accented_letters"],
    #         "language": hparams["language"],
    #         "skip_prep": hparams["skip_prep"],
    #     },
    # )
    

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    source_train_data, target_train_data, source_valid_data, target_valid_data, target_test_data, source_test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    logger.info(f"[MAIN] TRAINING")
    # Training
    with torch.autograd.detect_anomaly():
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            source_train_data,
            target_train_data,
            source_valid_data,
            target_valid_data,
            train_loader_kwargs=hparams["train_loader_kwargs"],
            valid_loader_kwargs=hparams["valid_loader_kwargs"],
        )

    # # Testing
    logger.info(f"[MAIN] TESTING TARGET")
    asr_brain.hparams.target_test_wer_file = hparams["target_test_wer_file"]
    asr_brain.evaluate(
        target_test_data,
        stage=Stage.TARGET_TEST,
        min_key="WER",
        test_loader_kwargs=hparams["test_loader_kwargs"],
    )
    
    logger.info(f"[MAIN] TESTING SOURCE")
    asr_brain.hparams.source_test_wer_file = hparams["source_test_wer_file"]
    asr_brain.evaluate(
        source_test_data,
        stage=Stage.SOURCE_TEST,
        min_key="WER",
        test_loader_kwargs=hparams["test_loader_kwargs"],
    )

    logger.info(f"[MAIN] SOURCE VALIDATING")
    asr_brain.hparams.source_valid_wer_file = hparams["source_valid_wer_file"]
    asr_brain.evaluate(
        source_valid_data,
        stage=Stage.VALID,
        min_key="WER",
        test_loader_kwargs=hparams["valid_loader_kwargs"],
    )
    
    logger.info(f"[MAIN] TARGET VALIDATING")
    asr_brain.hparams.target_valid_wer_file = hparams["target_valid_wer_file"]
    asr_brain.evaluate(
        target_valid_data,
        stage=Stage.VALID,
        min_key="WER",
        test_loader_kwargs=hparams["valid_loader_kwargs"],
    )