"""A de novo peptide sequencing model."""
#true model for PrimeNovo
import heapq
import threading
import logging
import re
import sys
import operator
import torch.nn.functional as F
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import depthcharge.masses
import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from . import mass_con
from ..components import ModelMixin, PeptideDecoder, SpectrumEncoder
from .ctc_beam_search import CTCBeamSearchDecoder
from . import evaluate
aa2mas = { 'G': 57.021464, 'A': 71.037114, 'S': 87.032028, 'P': 97.052764, 'V': 99.068414, 'T': 101.04767, 'C+57.021': 160.030649, 'L': 113.084064, 'I': 113.084064, 'N': 114.042927, 'D': 115.026943, 'Q': 128.058578, 'K': 128.094963, 'E': 129.042593, 'M': 131.040485, 'H': 137.058912, 'F': 147.068414, 'R': 156.101111, 'Y': 163.063329, 'W': 186.079313, 'M+15.995': 147.0354, 'N+0.984': 115.026943, 'Q+0.984': 129.042594, '+42.011': 42.010565, '+43.006': 43.005814, '-17.027': 100000, '+43.006-17.027': 25.980265, "_":0}

logger = logging.getLogger("casanovo")
def mass_cal(sequence):
    sequence = sequence.replace("I", "L")
    sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
    total = 0
    for each in sequence:
        
        # total += aa2mas[each]
        
        try:
            total += aa2mas[each]
        except:
            h1 = each.count("+42.011")
            h2 = each.count("+43.006")
            h3 = each.count("-17.027")
            total += h1 * 42.010565 + h2 * 43.005814  + h3 * -17.026549
            each = each.replace("+42.011", "")
            each = each.replace("+43.006", "")
            each = each.replace("-17.027", "")
            if each:
                total += aa2mas[each]
                    
    return total , sequence
def remove_repentance(index_list: List[int]) -> List[int]:
        """
        Eliminate repeated index in list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
        """
        return [a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b]
    
def ctc_post_processing( sentence_index: List[int]) -> List[int]:
        """
        Merge repetitive tokens, then eliminate <blank> tokens and <pad> tokens.
        The input sentence_index is expected to be a 1-D index list
        """
        sentence_index = remove_repentance(sentence_index)
        #sentence_index = list(filter((27).__ne__, sentence_index))
        temp = []
        for each in sentence_index:
            if each != 27:
                temp.append(each)
        # print("temp.", temp)
        return temp
     #sentence_index = list(filter((self.attn_decoder.get_pad_idx()).__ne__, sentence_index))

class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    custom_encoder : Optional[Union[SpectrumEncoder, PairedSpectrumEncoder]]
        A pretrained encoder to use. The ``dim_model`` of the encoder must be
        the same as that specified by the ``dim_model`` parameter here.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    n_beams: int
        Number of beams used during beam search decoding.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter: Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_writer: Optional[str]
        The output writer for the prediction results.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        custom_ctc_loss = True,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        custom_encoder: Optional[SpectrumEncoder] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        n_beams: int = 5,
        n_log: int = 10,
        mass_control_tol: float = 0.1,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter] = None,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_writer = None,
        ctc_dic: dict = {},
        PMC_enable = True,
        **kwargs: Dict,
    ):
        super().__init__()
        self.mass_control_tol = mass_control_tol
        self.save_hyperparameters()
        self.ctc_dic = ctc_dic
        self.PMC_enable = PMC_enable

        #assert PMC_enable == False
        self.ctc_dic["beam"] = n_beams
        
        
        # Build the model.
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=dim_intensity,
            )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
            max_pep_len = max_length

        )
        self.n_layers = n_layers
        #self.ctc_customized_mass_control = CTCMassControl(self.decoder )
        self.ctc_decoder = CTCBeamSearchDecoder(self.decoder, self.ctc_dic)
        self.calctime = 0.0
        #print("ctc_decoder:", self.ctc_decoder)
        '''
        decoding_params = {
                'force_length': getattr(args, 'force_length'),
                "desired_length": getattr(args, 'desired_length'),
                "use_length_ratio": getattr(args, 'use_length_ratio'),
                "k": getattr(args, 'k'),
                "beam_size": getattr(args, 'beam_size'),
                "scope": getattr(args, 'scope'),
                "marg_criteria": getattr(args, 'marg_criteria', 'max'),
                # truncate_summary is a dummy variable since length control does not need it
                "truncate_summary": getattr(args, 'truncate_summary'),
                "scaling_factor": getattr(args, 'bucket_size'),
            }
        '''
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.ctcloss = torch.nn.CTCLoss(blank = self.decoder.get_blank_idx(), zero_infinity=True) #to do
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs
        self.custom_ctc_loss = custom_ctc_loss

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.n_beams = n_beams
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues)
        #self.stop_token = self.decoder._aa2idx["$"] ## need to change 

        # Logging.
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer
    

    def forward(
            self, spectra: torch.Tensor,
            precursors: torch.Tensor, true_peps) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        peptides : List[List[str]]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        output_decoded_saved = []
        output_logits, _, output_list = self.decoder(None, precursors, *self.encoder(spectra))
        
        
        
        
        #explainitybility 
        
        # for each in output_list:
        #     top_tokensss, _ = self.ctc_decoder.decode(F.softmax(each, -1))
            
        #     output_decoded_saved.append(torch.flip(top_tokensss, dims= [-1]))
        # layer_pep_true = [[] for i in range( len(output_decoded_saved))]
        # layer_pep_pred = [[] for i in range( len(output_decoded_saved))]
        # for i in range( output_logits.shape[0]): # for each batch data
            
        #     for j in range( len(output_decoded_saved)) : #for each layer
        #         #print(output_decoded_saved[j][i])
                
        #         #print(i,"th data", j + 1, " th layer : ", "".join([self.decoder._idx2aa[each.item()] for each in output_decoded_saved[j][i] if each != -1 ]))
        #         #print("mass of above is: ", mass_cal("".join([self.decoder._idx2aa[each.item()] for each in output_decoded_saved[j][i] if each != -1 ]))[0])
        #         #print("jj", j )
        #         #print("len", len(layer_pep_pred))
        #         layer_pep_pred[j].append("".join([self.decoder._idx2aa[each.item()] for each in output_decoded_saved[j][i] if each != -1 ]))
        #         layer_pep_true[j].append(true_peps[i])
        # for i in range(len(output_decoded_saved)):
        #     aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
        #     *evaluate.aa_match_batch(layer_pep_pred[i], layer_pep_true[i],
        #                              self.decoder._peptide_mass.masses))
        #     log_args = dict(on_step=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        #     self.log("{} layer/aa_precision".format(i), aa_precision, **log_args)
        #     self.log("{} layer/aa_recall".format(i), aa_recall, **log_args)
        #     self.log("{} layer/pep_recall".format(i), pep_recall, **log_args)
            
        #     with open("./layer_samples.txt",'a') as f:
        #         f.write(str(int(i+1)) + ","+ str(float(aa_precision))+"\n")
            
            
        #----------explain_--------
        
        #output_logits, _, _ = self.decoder(None, precursors, *self.encoder(spectra))
        top_tokens, beamscores = self.ctc_decoder.decode(F.softmax(output_logits, -1))
        batchscores = beamscores.tolist()
        batchscores = 1 / torch.exp(beamscores)
        top_tokens_beam = top_tokens.tolist()
        #input_lengths = torch.full(size=(output_logits.size()[0],), fill_value=output_logits.size()[1])
        #top_tokens = self.ctc_customized_mass_control.decode(F.log_softmax(output_logits, -1),  precursors[:, 0])
        ''''''
        # --- here is multithreading-----
        batch_size = output_logits.shape[0]

        
        
        self.mass_offset_total = 0
        self.mass_offset_count = 0
        top_tokens = [[] for i in range(batch_size)]

        def worker (logits, mass, i , output_logits):
            #print("worker ", i , "started")
            mass = mass.clone().detach()
            if self.PMC_enable == False:
                '''
                sequence = list(filter((self.decoder.get_pad_idx()).__ne__, top_tokens_beam[i]))
                token_true = [self.decoder._idx2aa[each] for each in sequence]
                mass_true = mass[0].item() - 18.01
                mass_true_cal, seq = mass_cal("".join(token_true))
                if abs(mass_true_cal-mass_true) > 0.1:
                    print("not consistent here!!!!!!!,   " , abs(mass_true_cal-mass_true))
                    self.mass_offset_total+=abs(mass_true_cal-mass_true)
                    self.mass_offset_count+=1
                ''' 
                top_tokens[i] = top_tokens_beam[i]
                #print("skip ctc since mass greater than threshold")
                #sequence = list(filter((self.decoder.get_pad_idx()).__ne__, top_tokens_beam[i]))
                #token_true = [self.decoder._idx2aa[each] for each in sequence]
                #print("".join(reversed(token_true)))
            else:
                mass_true = mass[0].item() - 18.01
                #print(top_tokens_beam)
                sequence = list(filter((self.decoder.get_pad_idx()).__ne__, top_tokens_beam[i]))
                token_true = [self.decoder._idx2aa[each] for each in sequence]
                #if "+43.006" in "".join(token_true) or "+42.011" in "".join(token_true) or "-17.027" in "".join(token_true):
                if "hello" in "".join(token_true):
                    #ctc_customized_mass_control = CTCMassControl(self.decoder )
                    #top_tokens[i] = ctc_customized_mass_control.decode(logits, mass)[0]
                    temp = mass_con.knapDecode(logits, mass)
                    temp =  ctc_post_processing(temp)
                    if temp:
                        top_tokens[i] = temp
                        token_temp = [self.decoder._idx2aa[each] for each in temp]
                        token_temp = list(reversed(token_temp))
                        print("truth: ", token_true)
                        #print("truth: ", true_peps[i])
                        print("inf: " , token_temp)
                        # sys.stdout.flush()
                    else:
                        top_tokens[i] = top_tokens_beam[i]
                else: #will always execute
                #if True:
                    # correct mismatch mass, can only be used when true token is given
                    #----------------------
                    # cal_mass, seq= mass_cal(true_peps[i])
                    # if abs(cal_mass-  mass_true ) > 0.9:
                    #     mass[0] = mass[0] - 1
                    #     print("error after correction:", mass[0].item()-cal_mass, " ", )
                    #----------------------
                    #batchscores[i] = 1 / torch.exp(beamscores[i]) #commonted score
                    pred_mass, seq = mass_cal("".join(token_true))
                #print("True mass:", mass_true)
                #print("Decoded_mass: ", pred_mass)
                #print("aminos: ", seq)
                    # pred_mass = 0.0
                    assert self.mass_control_tol > 0
                    if abs(mass_true - pred_mass) < self.mass_control_tol: #if beam decoded peptide already less than tol, it's optimal path then, no need to call PMC
                        top_tokens[i] = top_tokens_beam[i]
                        # if beamscores[i] < 0.0:
                        #     print(beamscores[i])
                        #     print("I am CTCBeam")
                        # batchscores[i] = 1 / torch.exp(beamscores[i])
                        # if batchscores[i] > 1.0:
                        #     print("xianggebuxing:",beamscores[i])
                        
                        # print("beamscore:",beamscores[i])
                        #print("skip CTC length control")
                        
                    else:
                        #ctc_customized_mass_control = CTCMassControl(self.decoder )
                        # print("I am CUDA program")
                        temp = mass_con.knapDecode(logits, mass, self.mass_control_tol)
                        # knapscores = torch.exp(_)
                        # indTemp = torch.tensor(temp)
                        # knapscores = torch.softmax(output_logits,-1)[0]
                        # knapscores = knapscores[torch.arange(40),indTemp]
                        # scoreTemp = 1.0
                        # for x in knapscores:
                        #     scoreTemp *= x
                        # if scoreTemp == 0.0:
                        #     print("I am CUDA program")
                        #     print(knapscores)
                        # knapscores = scoreTemp
                        # knapscore = torch.sum(knapscores)
                        # knapscores = torch.exp(knapscore)
                        # print("knapscore:",torch.exp(_))
                        temp =  ctc_post_processing(temp)
                        if temp:
                            top_tokens[i] = temp
                            # batchscores[i] = knapscores
                            token_temp = [self.decoder._idx2aa[each] for each in temp]
                            token_temp = list(reversed(token_temp))
                            # print("truth: ", true_peps[i])
                            # print("inf: " , token_temp)
                            # sys.stdout.flush()
                        else:
                            top_tokens[i] = top_tokens_beam[i]
                            #batchscores[i] = 1 / torch.exp(beamscores[i]) #commented score
                        #sequence0 = list(filter((self.decoder.get_pad_idx()).__ne__, top_tokens[i]))
                        #token_true = [self.decoder._idx2aa[each] for each in sequence0]
                        #print("".join(reversed(token_true)))
            #print("woker ", i , "stopped")
        threads = []
        #log_prob = F.log_softmax(output_logits, -1)
        log_prob = F.log_softmax(output_logits, -1) #to change back
        for i in range(batch_size):
            t = threading.Thread(target=worker, args=(log_prob[[i], :, :], precursors[[i], 0], i, output_logits))
            threads.append(t)
            t.start()
        
        log_args = dict(on_step=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        #self.log("val/wrong_offset_average",self.mass_offset_total/self.mass_offset_count, **log_args)
        
        for t in threads:
            t.join()
        
        #-------------------
        #top_tokens = top_tokens.tolist()
        return [self.decoder.detokenize_truth(t, True) for t in top_tokens], batchscores

        '''
        aa_scores, tokens = self.beam_search_decode(  #to do 
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )
        '''
        #return [self.decoder.detokenize(t) for t in tokens], aa_scores

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        return self.decoder(sequences, precursors, *self.encoder(spectra))

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.

        """
        # import time

        # begintime=time.time()

        pred, truth, output_list = self._forward_step(*batch)
        assert len(output_list) == self.n_layers 
        

        # Compute AA recall, AA precision and Pep recall in training step
        # Author: Sheng Xu
        # Date: 20230220
        tokens = torch.argmax(pred, axis=2)
        peptides_pred = []
        peptides_true = []
        for idx in range(tokens.size()[0]):
            tokens_true = truth[idx,:]
            tokens_true = self.decoder.detokenize_truth(tokens_true)
            #print("tokens_true: ", ''.join(tokens_true))
            #print("tokens_batch: ", batch[2][idx])
            peptides_true.append(''.join(tokens_true))

            tokens_pred = tokens[idx,:]
            tokens_pred = self.decoder.detokenize(tokens_pred)
            peptides_pred.append(tokens_pred)
            #print("token_predict: ", ''.join(tokens_pred))

        aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(peptides_pred, peptides_true,
                                     self.decoder._peptide_mass.masses))
        
        rand = random.random()
        sampling_factor=0.8
        if(mode == "train"):
            sampling_factor=0.3
        if (rand < sampling_factor):
            peptides_pred_sample = []
            for tokenlist in peptides_pred:
                peptides_pred_sample.append(''.join(tokenlist))
            peptides_true_sample = peptides_true
            peptides_pair_list = list(zip(batch[1].cpu().numpy().tolist(),peptides_true_sample, peptides_pred_sample))
            peptides_pair = random.choices(peptides_pair_list, k=15)
            # print("peptides_pred",peptides_pred_sample)
            # print("peptides_true",peptides_true_sample)
            # print(aa_precision, aa_recall, pep_recall)
            if(not self.logger==None):
                self.logger.experiment[mode+"/peptides_pair"].append("Epoch: "+str(self.trainer.current_epoch)+str(peptides_pair))

            #self.log(mode+"/peptides_true", str(peptides_true_sample))
            #self.log(mode+"/peptides_pred", str(peptides_pred_sample))
        # endtime = time.time()
        # calctime = (endtime - begintime) / len(peptides_pred)
        
        if (mode == "train"):
            log_args = dict(on_step=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            self.log("train/aa_precision", aa_precision, **log_args)
            self.log("train/aa_recall", aa_recall, **log_args)
            self.log("train/pep_recall", pep_recall, **log_args)
        if (mode == "valid" and self.n_beams==0):
            log_args = dict(on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            # self.log("valid/spec/second", calctime, **log_args)
            self.log("valid/aa_precision", aa_precision, **log_args)
            self.log("valid/aa_recall", aa_recall, **log_args)
            self.log("valid/pep_recall", pep_recall, **log_args)
        if (mode == "test" and self.n_beams==0):
            log_args = dict(on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            self.log("test/aa_precision", aa_precision, **log_args)
            self.log("test/aa_recall", aa_recall, **log_args)
            self.log("test/pep_recall", pep_recall, **log_args)
            
        for idx, pred in enumerate(output_list):
            #if idx <= 0:  #5 
                #continue 
            total_loss = torch.tensor([0]).to(self.device)
            pred = pred.permute(1, 0, 2)
            
            input_lengths = torch.full(size=(pred.size()[1],), fill_value=pred.size()[0]) 
            target_lengths = (truth != self.decoder.get_pad_idx()).sum(axis=1) 
            if self.custom_ctc_loss == True :
                pass
            else:

                loss = self.ctcloss( torch.nn.functional.log_softmax(pred, dim=-1), truth, input_lengths, target_lengths ) 
                '''
                if self.trainer.current_epoch <= 50:
                    idx_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1]
                else:
                    idx_ratio = [ 0 for _ in range(self.n_layers-1)]
                    idx_ratio.append(1)
                '''
                #idx_ratio = [ 0 for _ in range(self.n_layers-1)]
                #idx_ratio.append(1)
                #idx_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1]
                #idx_ratio = [0.3, 0.5, 0.7, 0.8, 1, 1]
                idx_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1]
                #idx_ratio = [0, 0, 0, 0, 0.1, 0.1, 0,1, 0,1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1]
                idx_ratio = [ 0 for _ in range(self.n_layers-1)]
                idx_ratio.append(1)
                total_loss = total_loss + loss *idx_ratio[idx]
        #loss = self.celoss(pred, truth.flatten())
        loss = total_loss # to change
        
        if (mode == "train"):
            if(not self.logger==None):
                self.logger.experiment["train_CELoss_step"].append(loss)
            self.log(
                "train/CELoss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False
            )

        return loss

    def validation_step(self, batch, batch_idx=None, dataloader_idx=None) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        #print("validation ssssss")
        # import time

        # begintime = time.time()
        
        if (dataloader_idx==None): dataloader_idx=0
        key = "valid" if dataloader_idx==0 else "test"
        # Record the loss.
        loss = self.training_step(batch, mode=key)
        self.log(
            "valid/CELoss" if dataloader_idx==0 else "test/CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False
        )

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.

        if(self.n_beams>0):
            # print("Beam Search: 5")
            peptides_pred_raw, inferscores = self.forward(batch[0], batch[1], batch[2])
            # print("inferscores:",inferscores)
            # FIXME: Temporary fix to skip predictions with multiple stop tokens.
            peptides_pred, peptides_true = [], []
            for peptide_pred, peptide_true in zip(peptides_pred_raw, batch[2]):
                if len(peptide_pred) > 0:
                    if peptide_pred[0] == "$":
                        peptide_pred = peptide_pred[1:]  # Remove stop token.
                    if "$" not in peptide_pred and len(peptide_pred) > 0:
                        peptides_pred.append(peptide_pred)
                        peptides_true.append(peptide_true)
                        
            # with open("./denovo_random.txt",'a') as f:
            #     for i in range(len(peptides_pred_raw)):
                    
            #         print("label:",batch[2][i], ":" , peptides[i] , "\n")
                    
                        
            #         f.write("label: " + batch[2][i] + " prediction: " + "".join(peptides[i]) + " charge: " + str(int(batch[1][i][1])) +"\n")
                    
            #         f.write("label: " + batch[2][i] + " prediction : " + "".join(peptides_pred_raw[i]) + "\n")

            # AA_sim_sum = {"M+15.995" : 0.0000001, 'Q' : 0.0000001, 'F' : 0.0000001, "K": 0.0000001}
            # AA_sim_true = {"M+15.995" : 0, 'Q' : 0, 'F' : 0, "K": 0}
            # #---------xiang: write the scores out---------
            # for i in range(len(peptides_pred)):
            #     pred = peptides_pred[i]
            #     true = peptides_true[i]
            #     if isinstance(peptides_pred[i], str):
            #         pred = re.split(r"(?<=.)(?=[A-Z])", peptides_pred[i])
            #     if isinstance(peptides_true[i], str):
            #         true = re.split(r"(?<=.)(?=[A-Z])", peptides_true[i])
            #     matches , pep_match = evaluate.aa_match(pred,true,self.decoder._peptide_mass.masses)
            #     with open("./Vigna-mungo.txt",'a') as f:
            #         sum = np.sum(matches)
            #         aaPre = sum / len(pred)
            #         f.write(str(float(aaPre)) + " " + str(float(inferscores[i])) + " " + str(len(pred)) + "\n")
                    #f.write(str(float(pep_match)) + " " + str(float(inferscores[i])) + "\n")
            # #------------------------------

            #     pred_len = len(pred)
            #     true_len = len(true)
            #     min_len = min(pred_len, true_len)
            #     for x in range(min_len):
            #         if pred[x] in AA_sim_sum:
            #             AA_sim_sum[pred[x]] += 1
            #             if matches[x]:
            #                 AA_sim_true[pred[x]] += 1

            # endtime = time.time()
            # calctime = (endtime - begintime) / len(peptides_pred_raw)
            # for i in range(len(peptides_pred)):
            #     pred = peptides_pred[i]
            #     true = peptides_true[i]
            #     if isinstance(peptides_pred[i], str):
            #         pred = re.split(r"(?<=.)(?=[A-Z])", peptides_pred[i])
            #     if isinstance(peptides_true[i], str):
            #         true = re.split(r"(?<=.)(?=[A-Z])", peptides_true[i])

            #     matches,_ = evaluate.aa_match(pred,true,self.decoder._peptide_mass.masses)
            #     with open("./NATNovo/mouse.txt",'a') as f:
            #         sum = np.sum(matches)
            #         aaPre = sum / len(pred)
            #         f.write(str(float(aaPre)) + " " + str(float(inferscores[i])) + " " + str(len(pred)) + "\n")
                    
            aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
                *evaluate.aa_match_batch(peptides_pred, peptides_true,
                                         self.decoder._peptide_mass.masses))
            log_args = dict(on_step=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            # self.log("{}/spec/second".format(key), calctime, **log_args)
            # self.log("{}/M+0.984_precision".format(key), AA_sim_true["M+15.995"] / AA_sim_sum["M+15.995"], **log_args)
            # self.log("{}/Q_precision".format(key), AA_sim_true["Q"] / AA_sim_sum["Q"], **log_args)
            # self.log("{}/F_precision".format(key), AA_sim_true["F"] / AA_sim_sum["F"], **log_args)
            # self.log("{}/K_precision".format(key), AA_sim_true["K"] / AA_sim_sum["K"], **log_args)

            self.log("{}/aa_precision".format(key), aa_precision, **log_args)
            self.log("{}/aa_recall".format(key), aa_recall, **log_args)
            self.log("{}/pep_recall".format(key), pep_recall, **log_args)
            # print("{}/pep_recall".format(key), pep_recall)
            # sys.stdout.flush()
        return loss
    
    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        spectrum_idx : torch.Tensor
            The spectrum identifiers.
        precursors : torch.Tensor
            Precursor information for each spectrum.
        peptides : List[List[str]]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
  
        peptides , inferscores = self.forward(batch[0], batch[1], batch[2])
        import os
        
        file_path = "./denovo.tsv"
        headers = "label\tprediction\tcharge\tscore\n"

        # Check if the file exists and whether it contains headers
        if not os.path.exists(file_path) or open(file_path, 'r').readline().strip() != headers.strip():
            with open(file_path, 'a') as f:
                f.write(headers)

        # Append data
        with open(file_path,'a') as f:
            for i in range(len(peptides)):
                sequence = ""
                for el in peptides[i]:
                    if len(el) > 1:
                        if sequence == "" and (el[0] == '-' or el[0] == '+') :
                            sequence += '[' + el + ']-'
                        else:
                            sequence += el[0] + '[' + el[1:] + ']'
                    else:
                        sequence += el

                # print("label:",batch[2][i], ":" , peptides[i] , "\n")
                if batch[2][i].replace("$", "").replace("N+0.984", "D").replace("Q+0.984", "E").replace("L","I") == "".join(peptides[i]).replace("$", "").replace("N+0.984", "D").replace("Q+0.984", "E").replace("L","I"):
                    answer_is_correct = "correct"
                else:
                    answer_is_correct = "incorrect"
                #each line output this: label (title if label is none), predictions, charge, and confidence score
                f.write(batch[2][i].replace("\t", " ") + "\t" + sequence + "\t" + str(int(batch[1][i][1])) + "\t" + str(float(inferscores[i])) + "\n")
                
                #f.write("label: " + batch[2][i] + " prediction : " + "".join(peptides[i]) + "  " + answer_is_correct + "\n")
        
        return batch[2], batch[1], peptides  #batch[2]: identifier
    
    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train/CELoss"].detach()

        #self._history[-1]["train"] = train_loss
        #self._log_history()


    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        '''
        print("calculation:", self.calctime)
        callback_metrics = self.trainer.callback_metrics
        logger.info(callback_metrics)
        metrics = {
            "epoch": self.trainer.current_epoch,
            "valid": callback_metrics["valid/CELoss"].detach(),
            "valid_aa_precision":
            callback_metrics["valid/aa_precision"].detach(),
            "valid_aa_recall": callback_metrics["valid/aa_recall"].detach(),
            "valid_pep_recall": callback_metrics["valid/pep_recall"].detach(),
            "test": callback_metrics["test/CELoss"].detach(),
            "test_aa_precision":
            callback_metrics["test/aa_precision"].detach(),
            "test_aa_recall": callback_metrics["test/aa_recall"].detach(),
            "test_pep_recall": callback_metrics["test/pep_recall"].detach(),
        }
        self._history.append(metrics)
        self._log_history()
        '''

    def on_predict_epoch_end(
        self, results: List[List[Tuple[np.ndarray, List[str],
                                       torch.Tensor]]]) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """


        # if self.out_writer is None:
        #     return
        # for batch in results:
        #     for step in batch:
        #         for spectrum_i, precursor, aa_tokens in zip(*step):
        #             # Get peptide sequence, amino acid and peptide-level
        #             # confidence scores to write to output file.
        #             (
        #                 peptide,
        #                 aa_tokens,
        #                 peptide_score,
        #                 aa_scores,
        #             ) = self._get_output_peptide_and_scores(
        #                 aa_tokens, 1)
        #             # Compare the experimental vs calculated precursor m/z.
        #             _, precursor_charge, precursor_mz = precursor
        #             precursor_charge = int(precursor_charge.item())
        #             precursor_mz = precursor_mz.item()
        #             try:
        #                 calc_mz = self.peptide_mass_calculator.mass(
        #                     aa_tokens, precursor_charge)
        #                 delta_mass_ppm = [
        #                     _calc_mass_error(
        #                         calc_mz,
        #                         precursor_mz,
        #                         precursor_charge,
        #                         isotope,
        #                     ) for isotope in range(
        #                         self.isotope_error_range[0],
        #                         self.isotope_error_range[1] + 1,
        #                     )
        #                 ]
        #                 is_within_precursor_mz_tol = any(
        #                     abs(d) < self.precursor_mass_tol
        #                     for d in delta_mass_ppm)
        #             except KeyError:
        #                 calc_mz, is_within_precursor_mz_tol = np.nan, False
        #             # Subtract one if the precursor m/z tolerance is violated.
        #             if not is_within_precursor_mz_tol:
        #                 peptide_score -= 1

        #             self.out_writer.psms.append((
        #                 peptide,
        #                 spectrum_i,
        #                 peptide_score,
        #                 precursor_charge,
        #                 precursor_mz,
        #                 calc_mz,
        #                 aa_scores,
        #             ), )

    def _get_output_peptide_and_scores(
            self, aa_tokens: List[str],
            aa_scores: torch.Tensor) -> Tuple[str, List[str], float, str]:
        """
        Get peptide to output, amino acid and peptide-level confidence scores.

        Parameters
        ----------
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        aa_scores : torch.Tensor
            Amino acid-level confidence scores for the predicted sequence.

        Returns
        -------
        peptide : str
            Peptide sequence.
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        peptide_score : str
            Peptide-level confidence score.
        aa_scores : str
            Amino acid-level confidence scores for the predicted sequence.
        """
        # Omit stop token.
        aa_tokens = aa_tokens[1:] if self.decoder.reverse else aa_tokens[:-1]
        peptide = "".join(aa_tokens)

        # If this is a non-finished beam (after exceeding `max_length`), return
        # a dummy (empty) peptide and NaN scores.
        if len(peptide) == 0:
            aa_tokens = []

        # Take scores corresponding to the predicted amino acids. Reverse tokens
        # to correspond with correct amino acids as needed.
        step = -1 if self.decoder.reverse else 1
        top_aa_scores = [
            aa_score[self.decoder._aa2idx[aa_token]].item()
            for aa_score, aa_token in zip(aa_scores, aa_tokens[::step])
        ][::step]

        # Get peptide-level score from amino acid-level scores.
        peptide_score = _aa_to_pep_score(top_aa_scores)
        aa_scores = ",".join(list(map("{:.5f}".format, top_aa_scores)))
        return peptide, aa_tokens, peptide_score, aa_scores

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) > 0 and len(self._history[-1]) == 6:
            if len(self._history) == 1:
                logger.info(
                    "Epoch\tTrain loss\tValid loss\tAA precision\tAA recall\t"
                    "Peptide recall")
            metrics = self._history[-1]
            if metrics["epoch"] % self.n_log == 0:
                logger.info(
                    "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                    metrics["epoch"] + 1,
                    metrics.get("train", np.nan),
                    metrics.get("valid", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                    metrics.get("valid_aa_recall", np.nan),
                    metrics.get("valid_pep_recall", np.nan),
                )
                if self.tb_summarywriter is not None:
                    for descr, key in [
                        ("loss/train_crossentropy_loss", "train"),
                        ("loss/dev_crossentropy_loss", "valid"),
                        ("eval/dev_aa_precision", "valid_aa_precision"),
                        ("eval/dev_aa_recall", "valid_aa_recall"),
                        ("eval/dev_pep_recall", "valid_pep_recall"),
                    ]:
                        self.tb_summarywriter.add_scalar(
                            descr,
                            metrics.get(key, np.nan),
                            metrics["epoch"] + 1,
                        )

    def configure_optimizers(
        self, ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), **self.opt_kwargs)
        #optimizer = Lion(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.warmup_iters,
                                             max_iters=self.max_iters)
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        #add by xiang 
        '''
        if self.trainer.current_epoch >= 7:
            if total_norm >= 2.5:
                torch.nn.utils.clip_grad_norm(self.parameters(), 0.1)
        '''
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_before_clip"].append(total_norm)
    '''
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        print("hello!!!!!")

        print("val outputs: ", outputs )
        sys.stdout.flush()
    '''
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #print("hello outputs??: ", outputs)
        # sys.stdout.flush()
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_after_clip"].append(total_norm)
            
    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int,
                 max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):

        # Cosine annealing after a constant period
        # Author: Sheng Xu
        # Date: 20230214

        decay=self.warmup/self.max_iters
        if epoch <= self.warmup and self.warmup>0:
            #lr_factor = 1
            
            lr_factor = 1 * (epoch / self.warmup)
        else:
            
            lr_factor = 0.5 * (1 + np.cos(np.pi * ( (epoch - (decay * self.max_iters)) / ((1-decay) * self.max_iters))))
            #lr_factor = 0.05

        return lr_factor

def _aa_to_pep_score(aa_scores: List[float]) -> float:
    """
    Calculate peptide-level confidence score from amino acid level scores.

    Parameters
    ----------
    aa_scores : List[float]
        Amino acid level confidence scores.

    Returns
    -------
    float
        Peptide confidence score.
    """
    return np.mean(aa_scores)
def _calc_mass_error(calc_mz: float,
                     obs_mz: float,
                     charge: int,
                     isotope: int = 0) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


