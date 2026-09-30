"""Generated reference surfaces for upstream PottsMPNN and LASErMPNN knobs.

Produced by scripts/redsox/extract_upstream_knobs.py. Do not edit the
generated dataclasses by hand. The manual block is the exception.
"""

from __future__ import annotations

from dataclasses import dataclass

POTTSMPNN_COMMIT = "0cb0a58874e373d664114f1735deab77614e4ab5"
LASERMPNN_COMMIT = "e70f2c6d765416f7e29d51bfd6d4e08496438878"
EXTRACTOR_SHA256 = "ed93a91e897ba64ad55fe1f3b7a6c425f9b11bf7260b49332007b5f639eb11c7"

@dataclass(frozen=True)
class LaserCheckpointParamsKnobs:
  """model_params build_hydrogens / graph_structure literals and subscript reads."""

  laser_checkpoint__build_hydrogens: bool = True  # type=checkpoint literal; help=checkpoint-derived model_params key; train_lasermpnn.py:779
  laser_checkpoint__graph_structure__lig_lig_knn_graph_k: int = 5  # type=checkpoint literal; help=checkpoint-derived model_params key; pretrain_ligand_encoder.py:227
  laser_checkpoint__graph_structure__lig_pr_distance_cutoff: float = 20.0  # type=checkpoint literal; help=checkpoint-derived model_params key; train_lasermpnn.py:794
  laser_checkpoint__graph_structure__lig_pr_knn_graph_k: int = 48  # type=checkpoint literal; help=checkpoint-derived model_params key; train_lasermpnn.py:795
  laser_checkpoint__graph_structure__pr_pr_knn_graph_k: int = 48  # type=checkpoint literal; help=checkpoint-derived model_params key; train_lasermpnn.py:793

@dataclass(frozen=True)
class LaserModelLASErMPNNSampleKnobs:
  """utils/model.py:727:sample."""

  laser_model_LASErMPNN_sample__ala_budget: int = 4  # type=int; help=parameter of sample; utils/model.py:729
  laser_model_LASErMPNN_sample__batch: object | None = None  # type=BatchData; help=parameter of sample; utils/model.py:728; UNRESOLVED required parameter; no upstream default
  laser_model_LASErMPNN_sample__budget_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of sample; utils/model.py:729
  laser_model_LASErMPNN_sample__chi_angle_sample_temperature: object | None = None  # type=Optional[float]; help=parameter of sample; utils/model.py:731
  laser_model_LASErMPNN_sample__chi_min_p: float = 0.0  # type=float; help=parameter of sample; utils/model.py:732
  laser_model_LASErMPNN_sample__disable_charged_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of sample; utils/model.py:733
  laser_model_LASErMPNN_sample__disable_pbar: bool = False  # type=bool; help=parameter of sample; utils/model.py:732
  laser_model_LASErMPNN_sample__disabled_residues: tuple[object, ...] = ('X',)  # type=Optional[list]; help=parameter of sample; utils/model.py:731
  laser_model_LASErMPNN_sample__gly_budget: int = 0  # type=int; help=parameter of sample; utils/model.py:729
  laser_model_LASErMPNN_sample__ignore_chain_mask_zeros: bool = False  # type=bool; help=parameter of sample; utils/model.py:733
  laser_model_LASErMPNN_sample__repack_all: bool = False  # type=bool; help=parameter of sample; utils/model.py:733
  laser_model_LASErMPNN_sample__return_encoder_embeddings: bool = False  # type=bool; help=parameter of sample; utils/model.py:732
  laser_model_LASErMPNN_sample__seq_min_p: float = 0.0  # type=float; help=parameter of sample; utils/model.py:732
  laser_model_LASErMPNN_sample__sequence_sample_temperature: object | None = None  # type=Optional[Union[float, torch.Tensor]]; help=parameter of sample; utils/model.py:730

@dataclass(frozen=True)
class LaserModelLASErMPNNTiedSampleKnobs:
  """utils/model.py:460:tied_sample."""

  laser_model_LASErMPNN_tied_sample__ala_budget: int = 4  # type=int; help=parameter of tied_sample; utils/model.py:462
  laser_model_LASErMPNN_tied_sample__batch1: object | None = None  # type=BatchData; help=parameter of tied_sample; utils/model.py:461; UNRESOLVED required parameter; no upstream default
  laser_model_LASErMPNN_tied_sample__batch2: object | None = None  # type=BatchData; help=parameter of tied_sample; utils/model.py:461; UNRESOLVED required parameter; no upstream default
  laser_model_LASErMPNN_tied_sample__budget_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of tied_sample; utils/model.py:462
  laser_model_LASErMPNN_tied_sample__chi_angle_sample_temperature: object | None = None  # type=Optional[float]; help=parameter of tied_sample; utils/model.py:463
  laser_model_LASErMPNN_tied_sample__disable_pbar: bool = False  # type=bool; help=parameter of tied_sample; utils/model.py:464
  laser_model_LASErMPNN_tied_sample__disabled_residues: tuple[object, ...] = ('X',)  # type=Optional[list]; help=parameter of tied_sample; utils/model.py:463
  laser_model_LASErMPNN_tied_sample__gly_budget: int = 0  # type=int; help=parameter of tied_sample; utils/model.py:462
  laser_model_LASErMPNN_tied_sample__lambda_: float = 0.5  # type=float; help=parameter of tied_sample; utils/model.py:461
  laser_model_LASErMPNN_tied_sample__repack_all: bool = False  # type=bool; help=parameter of tied_sample; utils/model.py:464
  laser_model_LASErMPNN_tied_sample__sequence_sample_temperature: object | None = None  # type=Optional[Union[float, torch.Tensor]]; help=parameter of tied_sample; utils/model.py:461

@dataclass(frozen=True)
class LaserModelLigandmpnnLigandMPNNSampleKnobs:
  """utils/model_ligandmpnn.py:241:sample."""

  laser_model_ligandmpnn_LigandMPNN_sample__batch: object | None = None  # type=BatchData; help=parameter of sample; utils/model_ligandmpnn.py:242; UNRESOLVED required parameter; no upstream default
  laser_model_ligandmpnn_LigandMPNN_sample__disable_charged_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of sample; utils/model_ligandmpnn.py:245
  laser_model_ligandmpnn_LigandMPNN_sample__disable_pbar: bool = False  # type=bool; help=parameter of sample; utils/model_ligandmpnn.py:244
  laser_model_ligandmpnn_LigandMPNN_sample__disabled_residues: tuple[object, ...] = ('X',)  # type=Optional[list]; help=parameter of sample; utils/model_ligandmpnn.py:243
  laser_model_ligandmpnn_LigandMPNN_sample__ignore_chain_mask_zeros: bool = False  # type=bool; help=parameter of sample; utils/model_ligandmpnn.py:245
  laser_model_ligandmpnn_LigandMPNN_sample__repack_all: bool = False  # type=bool; help=parameter of sample; utils/model_ligandmpnn.py:245
  laser_model_ligandmpnn_LigandMPNN_sample__return_encoder_embeddings: bool = False  # type=bool; help=parameter of sample; utils/model_ligandmpnn.py:244
  laser_model_ligandmpnn_LigandMPNN_sample__seq_min_p: float = 0.0  # type=float; help=parameter of sample; utils/model_ligandmpnn.py:244
  laser_model_ligandmpnn_LigandMPNN_sample__sequence_sample_temperature: object | None = None  # type=Optional[Union[float, torch.Tensor]]; help=parameter of sample; utils/model_ligandmpnn.py:242

@dataclass(frozen=True)
class LaserRunBatchInferenceKnobs:
  """run_batch_inference.py."""

  laser_run_batch_inference__ala_budget: int = 4  # type=int; help=Maximum number of ALA residues that can be sampled in exposed non-secondary structured residues. Only used if --constrain_ala_gly_sampling_to_exposed_non_secondary_structure is set or --budget_residue_sele_string is set.; run_batch_inference.py:386
  laser_run_batch_inference__budget_residue_sele_string: object | None = None  # type=unknown; help=A ProDy-style selection string to constrain the residues to sample ALA and GLY residues in. Can be used to override the automatic selection performed by the --constrain_ala_gly_sampling_to_exposed_non_secondary_structure flag.; run_batch_inference.py:385
  laser_run_batch_inference__chi_min_p: float = 0.0  # type=float; help=Minimum probability for chi sampling. Not recommended.; run_batch_inference.py:373
  laser_run_batch_inference__chi_temp: float | None = None  # type=float; help=Temperature for chi sampling.; run_batch_inference.py:372
  laser_run_batch_inference__constrain_ala_gly_sampling_to_exposed_non_secondary_structure: bool = False  # type=bool; help=If set, constrains number of ALA and GLY residues that can be sampled in exposed non-secondary structured residues. The max number of ALA and GLY residues is set by the --ala_budget and --gly_budget arguments.; run_batch_inference.py:384
  laser_run_batch_inference__designs_per_batch: int = 30  # type=int; help=Number of designs to generate per batch. If designs_per_input > designs_per_batch, chunks up the inference calls in batches of this size. Default is 30, can increase/decrease depending on available GPU memory.; run_batch_inference.py:366
  laser_run_batch_inference__designs_per_input: int | None = None  # type=int; help=Number of designs to generate per input.; run_batch_inference.py:365; UNRESOLVED no default literal
  laser_run_batch_inference__disable_charged_fs: bool = False  # type=bool; help=Disable sampling D,K,R,E residues in the first shell around the ligand.; run_batch_inference.py:397
  laser_run_batch_inference__disabled_residues: str = 'X,C'  # type=str; help=Residues to disable in sampling.; run_batch_inference.py:380
  laser_run_batch_inference__first_shell_sequence_temp: float | None = None  # type=float; help=Temperature for first shell sequence sampling. Can be used to disentangle binding site temperature from global sequence temperature for harder folds.; run_batch_inference.py:371
  laser_run_batch_inference__fix_beta: bool = False  # type=bool; help=If B-factors are set to 1, fixes the residue and rotamer, if not, designs that position.; run_batch_inference.py:381
  laser_run_batch_inference__fs_calc_burial_hull_alpha_value: float = 9.0  # type=float; help=Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).; run_batch_inference.py:395
  laser_run_batch_inference__fs_calc_ca_distance: float = 10.0  # type=float; help=Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.; run_batch_inference.py:394
  laser_run_batch_inference__fs_no_calc_burial: bool = False  # type=bool; help=Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance; run_batch_inference.py:396
  laser_run_batch_inference__gly_budget: int = 0  # type=int; help=Maximum number of GLY residues that can be sampled in exposed non-secondary structured residues. Only used if --constrain_ala_gly_sampling_to_exposed_non_secondary_structure is set or --budget_residue_sele_string is set.; run_batch_inference.py:387
  laser_run_batch_inference__ignore_key_mismatch: bool = True  # type=bool; help=Allows mismatched keys in checkpoint statedict; run_batch_inference.py:379
  laser_run_batch_inference__ignore_ligand: bool = False  # type=bool; help=Ignore ligand in sampling.; run_batch_inference.py:383
  laser_run_batch_inference__inference_device: str = 'cpu'  # type=str; help=PyTorch style device string (e.g. "cuda:0").; run_batch_inference.py:376
  laser_run_batch_inference__input_pdb_directory: str | None = None  # type=str; help=Path to directory of input .pdb or .pdb.gz files, a single input .pdb or .pdb.gz file, or a .txt file of paths to input .pdb or .pdb.gz files.; run_batch_inference.py:363; UNRESOLVED no default literal
  laser_run_batch_inference__inputs_processed_simultaneously: int = 5  # type=int; help=When passed a list of multiple files, this is the number of input files to process per pass through the GPU. Useful when generating a few sequences for many input files.; run_batch_inference.py:367
  laser_run_batch_inference__model_weights_path: str | None = None  # type=str; help=f'Path to model weights. Default: {default_weights_path}. Other weights can be found in the ./model_weights/ directory.'; run_batch_inference.py:368; UNRESOLVED default is not a literal (f'{default_weights_path}')
  laser_run_batch_inference__noncanonical_aa_ligand: bool = False  # type=bool; help=Featurize a noncanonical amino acid as a ligand.; run_batch_inference.py:388
  laser_run_batch_inference__output_fasta: bool = False  # type=bool; help=Output a fasta file of the designed sequences in addition to the PDB files.; run_batch_inference.py:391
  laser_run_batch_inference__output_fasta_only: bool = False  # type=bool; help=Output only a fasta file of the designed sequences, does not write PDB files.; run_batch_inference.py:392
  laser_run_batch_inference__output_pdb_directory: str | None = None  # type=str; help=Path to directory to output LASErMPNN designs.; run_batch_inference.py:364; UNRESOLVED no default literal
  laser_run_batch_inference__repack_all: bool = False  # type=bool; help=Repack all residues, even those with chain_mask=1.; run_batch_inference.py:389
  laser_run_batch_inference__repack_only_input_sequence: bool = False  # type=bool; help=Repacks the input sequence without changing the sequence.; run_batch_inference.py:382
  laser_run_batch_inference__seq_min_p: float = 0.0  # type=float; help=Minimum probability for sequence sampling. Not recommended.; run_batch_inference.py:374
  laser_run_batch_inference__sequence_temp: float | None = None  # type=float; help=Temperature for sequence sampling.; run_batch_inference.py:370
  laser_run_batch_inference__use_water: bool = False  # type=bool; help=Parses water (resname HOH) as part of a ligand.; run_batch_inference.py:377
  laser_run_batch_inference__verbose: bool = True  # type=bool; help=Silences all output except pbar.; run_batch_inference.py:378

@dataclass(frozen=True)
class LaserRunBatchInferenceLigandmpnnKnobs:
  """run_batch_inference_ligandmpnn.py."""

  laser_run_batch_inference_ligandmpnn__ala_budget: int = 4  # type=int; help=; run_batch_inference_ligandmpnn.py:330
  laser_run_batch_inference_ligandmpnn__budget_residue_sele_string: object | None = None  # type=unknown; help=; run_batch_inference_ligandmpnn.py:329
  laser_run_batch_inference_ligandmpnn__chi_min_p: float = 0.0  # type=float; help=Minimum probability for chi sampling. Not recommended.; run_batch_inference_ligandmpnn.py:318
  laser_run_batch_inference_ligandmpnn__chi_temp: float | None = None  # type=float; help=Temperature for chi sampling.; run_batch_inference_ligandmpnn.py:317
  laser_run_batch_inference_ligandmpnn__designs_per_batch: int = 30  # type=int; help=Number of designs to generate per batch. If designs_per_input > designs_per_batch, chunks up the inference calls in batches of this size. Default is 30, can increase/decrease depending on available GPU memory.; run_batch_inference_ligandmpnn.py:311
  laser_run_batch_inference_ligandmpnn__designs_per_input: int | None = None  # type=int; help=Number of designs to generate per input.; run_batch_inference_ligandmpnn.py:310; UNRESOLVED no default literal
  laser_run_batch_inference_ligandmpnn__disable_charged_fs: bool = False  # type=bool; help=Disable sampling D,K,R,E residues in the first shell around the ligand.; run_batch_inference_ligandmpnn.py:341
  laser_run_batch_inference_ligandmpnn__disabled_residues: str = 'X'  # type=str; help=Residues to disable in sampling.; run_batch_inference_ligandmpnn.py:325
  laser_run_batch_inference_ligandmpnn__first_shell_sequence_temp: float | None = None  # type=float; help=Temperature for first shell sequence sampling. Can be used to disentangle binding site temperature from global sequence temperature for harder folds.; run_batch_inference_ligandmpnn.py:316
  laser_run_batch_inference_ligandmpnn__fix_beta: bool = False  # type=bool; help=If B-factors are set to 1, fixes the residue and rotamer, if not, designs that position.; run_batch_inference_ligandmpnn.py:326
  laser_run_batch_inference_ligandmpnn__fs_calc_burial_hull_alpha_value: float = 9.0  # type=float; help=Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).; run_batch_inference_ligandmpnn.py:339
  laser_run_batch_inference_ligandmpnn__fs_calc_ca_distance: float = 10.0  # type=float; help=Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.; run_batch_inference_ligandmpnn.py:338
  laser_run_batch_inference_ligandmpnn__fs_no_calc_burial: bool = False  # type=bool; help=Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance; run_batch_inference_ligandmpnn.py:340
  laser_run_batch_inference_ligandmpnn__gly_budget: int = 0  # type=int; help=; run_batch_inference_ligandmpnn.py:331
  laser_run_batch_inference_ligandmpnn__ignore_key_mismatch: bool = True  # type=bool; help=Allows mismatched keys in checkpoint statedict; run_batch_inference_ligandmpnn.py:324
  laser_run_batch_inference_ligandmpnn__ignore_ligand: bool = False  # type=bool; help=Ignore ligand in sampling.; run_batch_inference_ligandmpnn.py:328
  laser_run_batch_inference_ligandmpnn__inference_device: str = 'cpu'  # type=str; help=PyTorch style device string (e.g. "cuda:0").; run_batch_inference_ligandmpnn.py:321
  laser_run_batch_inference_ligandmpnn__input_pdb_directory: str | None = None  # type=str; help=Path to directory of input .pdb or .pdb.gz files, a single input .pdb or .pdb.gz file, or a .txt file of paths to input .pdb or .pdb.gz files.; run_batch_inference_ligandmpnn.py:308; UNRESOLVED no default literal
  laser_run_batch_inference_ligandmpnn__inputs_processed_simultaneously: int = 5  # type=int; help=When passed a list of multiple files, this is the number of input files to process per pass through the GPU. Useful when generating a few sequences for many input files.; run_batch_inference_ligandmpnn.py:312
  laser_run_batch_inference_ligandmpnn__model_weights_path: str | None = None  # type=str; help=f'Path to model weights. Default: {default_weights_path}. Other weights can be found in the ./model_weights/ directory.'; run_batch_inference_ligandmpnn.py:313; UNRESOLVED default is not a literal (f'{default_weights_path}')
  laser_run_batch_inference_ligandmpnn__noncanonical_aa_ligand: bool = False  # type=bool; help=Featurize a noncanonical amino acid as a ligand.; run_batch_inference_ligandmpnn.py:332
  laser_run_batch_inference_ligandmpnn__output_fasta: bool = False  # type=bool; help=Output a fasta file of the designed sequences in addition to the PDB files.; run_batch_inference_ligandmpnn.py:335
  laser_run_batch_inference_ligandmpnn__output_fasta_only: bool = False  # type=bool; help=Output only a fasta file of the designed sequences, does not write PDB files.; run_batch_inference_ligandmpnn.py:336
  laser_run_batch_inference_ligandmpnn__output_pdb_directory: str | None = None  # type=str; help=Path to directory to output LASErMPNN designs.; run_batch_inference_ligandmpnn.py:309; UNRESOLVED no default literal
  laser_run_batch_inference_ligandmpnn__repack_all: bool = False  # type=bool; help=Repack all residues, even those with chain_mask=1.; run_batch_inference_ligandmpnn.py:333
  laser_run_batch_inference_ligandmpnn__repack_only_input_sequence: bool = False  # type=bool; help=Repacks the input sequence without changing the sequence.; run_batch_inference_ligandmpnn.py:327
  laser_run_batch_inference_ligandmpnn__seq_min_p: float = 0.0  # type=float; help=Minimum probability for sequence sampling. Not recommended.; run_batch_inference_ligandmpnn.py:319
  laser_run_batch_inference_ligandmpnn__sequence_temp: float | None = None  # type=float; help=Temperature for sequence sampling.; run_batch_inference_ligandmpnn.py:315
  laser_run_batch_inference_ligandmpnn__use_water: bool = False  # type=bool; help=Parses water (resname HOH) as part of a ligand.; run_batch_inference_ligandmpnn.py:322
  laser_run_batch_inference_ligandmpnn__verbose: bool = True  # type=bool; help=Silences all output except pbar.; run_batch_inference_ligandmpnn.py:323

@dataclass(frozen=True)
class LaserRunInferenceKnobs:
  """run_inference.py."""

  laser_run_inference__backbone_noise: str = ''  # type=str; help=Inference backbone noise.; run_inference.py:779
  laser_run_inference__device: str = 'cpu'  # type=str; help=Pytorch style device string. Ex: "cuda:0" or "cpu".; run_inference.py:780
  laser_run_inference__disable_charged_fs: bool = False  # type=bool; help=Disable sampling D,K,R,E residues in the first shell around the ligand.; run_inference.py:792
  laser_run_inference__entropy_decoder: bool = False  # type=bool; help=Uses entropy based decoding order. Decodes all residues and selects the lowest entropy residue as next to decode, then recomputes all remaining residues. Takes longer than normal decoding.; run_inference.py:784
  laser_run_inference__fix_beta: bool = False  # type=bool; help=Residues with B-Factor of 1.0 have sequence and rotamer fixed, residues with B-Factor of 0.0 are designed.; run_inference.py:782
  laser_run_inference__fs_calc_burial_hull_alpha_value: float = 9.0  # type=float; help=Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).; run_inference.py:790
  laser_run_inference__fs_calc_ca_distance: float = 10.0  # type=float; help=Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.; run_inference.py:789
  laser_run_inference__fs_no_calc_burial: bool = False  # type=bool; help=Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance; run_inference.py:791
  laser_run_inference__fs_sequence_temp: float | None = None  # type=float; help=Residues around the ligand will be sampled at this temperature, otherwise they default to sequence_temp.; run_inference.py:778
  laser_run_inference__ignore_ligand: bool = False  # type=bool; help=Ignore ligands in the input PDB file.; run_inference.py:786
  laser_run_inference__input_pdb_code: str | None = None  # type=str; help=Path to the input PDB file.; run_inference.py:773; UNRESOLVED no default literal
  laser_run_inference__model_weights: str | None = None  # type=str; help=f'Path to dictionary of torch.save()ed model state_dict and training parameters. Default: {default_weights_path}'; run_inference.py:774; UNRESOLVED default is not a literal (default_weights_path)
  laser_run_inference__noncanonical_aa_ligand: bool = False  # type=bool; help=Featurize a noncanonical amino acid as a ligand.; run_inference.py:787
  laser_run_inference__output_path: str = 'laser_output.pdb'  # type=str; help=Path to the output PDB file.; run_inference.py:776
  laser_run_inference__repack_only: bool = False  # type=bool; help=Only repack residues, do not design new ones.; run_inference.py:785
  laser_run_inference__sequence_temp: str = ''  # type=str; help=Sequence sample temperature.; run_inference.py:777
  laser_run_inference__strict_load: bool = True  # type=bool; help=Small state_dict mismatches are ignored. Don't use this unless any missing parameters aren't learned during training.; run_inference.py:783

@dataclass(frozen=True)
class LaserRunInferenceLigandmpnnKnobs:
  """run_inference_ligandmpnn.py."""

  laser_run_inference_ligandmpnn__backbone_noise: str = ''  # type=str; help=Inference backbone noise.; run_inference_ligandmpnn.py:777
  laser_run_inference_ligandmpnn__device: str = 'cpu'  # type=str; help=Pytorch style device string. Ex: "cuda:0" or "cpu".; run_inference_ligandmpnn.py:778
  laser_run_inference_ligandmpnn__disable_charged_fs: bool = False  # type=bool; help=Disable sampling D,K,R,E residues in the first shell around the ligand.; run_inference_ligandmpnn.py:790
  laser_run_inference_ligandmpnn__entropy_decoder: bool = False  # type=bool; help=Uses entropy based decoding order. Decodes all residues and selects the lowest entropy residue as next to decode, then recomputes all remaining residues. Takes longer than normal decoding.; run_inference_ligandmpnn.py:782
  laser_run_inference_ligandmpnn__fix_beta: bool = False  # type=bool; help=Residues with B-Factor of 1.0 have sequence and rotamer fixed, residues with B-Factor of 0.0 are designed.; run_inference_ligandmpnn.py:780
  laser_run_inference_ligandmpnn__fs_calc_burial_hull_alpha_value: float = 9.0  # type=float; help=Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).; run_inference_ligandmpnn.py:788
  laser_run_inference_ligandmpnn__fs_calc_ca_distance: float = 10.0  # type=float; help=Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.; run_inference_ligandmpnn.py:787
  laser_run_inference_ligandmpnn__fs_no_calc_burial: bool = False  # type=bool; help=Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance; run_inference_ligandmpnn.py:789
  laser_run_inference_ligandmpnn__fs_sequence_temp: float | None = None  # type=float; help=Residues around the ligand will be sampled at this temperature, otherwise they default to sequence_temp.; run_inference_ligandmpnn.py:776
  laser_run_inference_ligandmpnn__ignore_ligand: bool = False  # type=bool; help=Ignore ligands in the input PDB file.; run_inference_ligandmpnn.py:784
  laser_run_inference_ligandmpnn__input_pdb_code: str | None = None  # type=str; help=Path to the input PDB file.; run_inference_ligandmpnn.py:771; UNRESOLVED no default literal
  laser_run_inference_ligandmpnn__model_weights: str | None = None  # type=str; help=f'Path to dictionary of torch.save()ed model state_dict and training parameters. Default: {default_weights_path}'; run_inference_ligandmpnn.py:772; UNRESOLVED default is not a literal (default_weights_path)
  laser_run_inference_ligandmpnn__noncanonical_aa_ligand: bool = False  # type=bool; help=Featurize a noncanonical amino acid as a ligand.; run_inference_ligandmpnn.py:785
  laser_run_inference_ligandmpnn__output_path: str = 'laser_output.pdb'  # type=str; help=Path to the output PDB file.; run_inference_ligandmpnn.py:774
  laser_run_inference_ligandmpnn__repack_only: bool = False  # type=bool; help=Only repack residues, do not design new ones.; run_inference_ligandmpnn.py:783
  laser_run_inference_ligandmpnn__sequence_temp: str = ''  # type=str; help=Sequence sample temperature.; run_inference_ligandmpnn.py:775
  laser_run_inference_ligandmpnn__strict_load: bool = True  # type=bool; help=Small state_dict mismatches are ignored. Don't use this unless any missing parameters aren't learned during training.; run_inference_ligandmpnn.py:781

@dataclass(frozen=True)
class LaserRunInferenceLigandmpnnSampleModelKnobs:
  """run_inference_ligandmpnn.py:512:sample_model."""

  laser_run_inference_ligandmpnn_sample_model__ala_budget: int = 4  # type=Optional[int]; help=parameter of sample_model; run_inference_ligandmpnn.py:516
  laser_run_inference_ligandmpnn_sample_model__batch_data: object | None = None  # type=BatchData; help=parameter of sample_model; run_inference_ligandmpnn.py:513; UNRESOLVED required parameter; no upstream default
  laser_run_inference_ligandmpnn_sample_model__bb_noise: float | None = None  # type=float; help=parameter of sample_model; run_inference_ligandmpnn.py:513; UNRESOLVED required parameter; no upstream default
  laser_run_inference_ligandmpnn_sample_model__budget_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of sample_model; run_inference_ligandmpnn.py:516
  laser_run_inference_ligandmpnn_sample_model__chi_min_p: float = 0.0  # type=float; help=parameter of sample_model; run_inference_ligandmpnn.py:514
  laser_run_inference_ligandmpnn_sample_model__chi_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference_ligandmpnn.py:513
  laser_run_inference_ligandmpnn_sample_model__disable_charged_fs: bool = False  # type=bool; help=parameter of sample_model; run_inference_ligandmpnn.py:517
  laser_run_inference_ligandmpnn_sample_model__disable_pbar: bool = False  # type=bool; help=parameter of sample_model; run_inference_ligandmpnn.py:514
  laser_run_inference_ligandmpnn_sample_model__disabled_residues: tuple[object, ...] = ('X',)  # type=Optional[List[str]]; help=parameter of sample_model; run_inference_ligandmpnn.py:515
  laser_run_inference_ligandmpnn_sample_model__fs_sequence_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference_ligandmpnn.py:514
  laser_run_inference_ligandmpnn_sample_model__gly_budget: int = 0  # type=Optional[int]; help=parameter of sample_model; run_inference_ligandmpnn.py:516
  laser_run_inference_ligandmpnn_sample_model__ignore_chain_mask_zeros: bool = False  # type=bool; help=parameter of sample_model; run_inference_ligandmpnn.py:515
  laser_run_inference_ligandmpnn_sample_model__model: object | None = None  # type=LigandMPNN; help=parameter of sample_model; run_inference_ligandmpnn.py:513; UNRESOLVED required parameter; no upstream default
  laser_run_inference_ligandmpnn_sample_model__params: object | None = None  # type=dict; help=parameter of sample_model; run_inference_ligandmpnn.py:513; UNRESOLVED required parameter; no upstream default
  laser_run_inference_ligandmpnn_sample_model__repack_all: bool = False  # type=bool; help=parameter of sample_model; run_inference_ligandmpnn.py:515
  laser_run_inference_ligandmpnn_sample_model__seq_min_p: float = 0.0  # type=float; help=parameter of sample_model; run_inference_ligandmpnn.py:514
  laser_run_inference_ligandmpnn_sample_model__sequence_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference_ligandmpnn.py:513; UNRESOLVED required parameter; no upstream default
  laser_run_inference_ligandmpnn_sample_model__use_edo: bool = False  # type=bool; help=parameter of sample_model; run_inference_ligandmpnn.py:513
  laser_run_inference_ligandmpnn_sample_model__verbose: bool = False  # type=bool; help=parameter of sample_model; run_inference_ligandmpnn.py:514

@dataclass(frozen=True)
class LaserRunInferenceSampleModelKnobs:
  """run_inference.py:511:sample_model."""

  laser_run_inference_sample_model__ala_budget: int = 4  # type=Optional[int]; help=parameter of sample_model; run_inference.py:515
  laser_run_inference_sample_model__batch_data: object | None = None  # type=BatchData; help=parameter of sample_model; run_inference.py:512; UNRESOLVED required parameter; no upstream default
  laser_run_inference_sample_model__bb_noise: float | None = None  # type=float; help=parameter of sample_model; run_inference.py:512; UNRESOLVED required parameter; no upstream default
  laser_run_inference_sample_model__budget_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of sample_model; run_inference.py:515
  laser_run_inference_sample_model__chi_min_p: float = 0.0  # type=float; help=parameter of sample_model; run_inference.py:513
  laser_run_inference_sample_model__chi_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference.py:512
  laser_run_inference_sample_model__disable_charged_fs: bool = False  # type=bool; help=parameter of sample_model; run_inference.py:516
  laser_run_inference_sample_model__disable_pbar: bool = False  # type=bool; help=parameter of sample_model; run_inference.py:513
  laser_run_inference_sample_model__disabled_residues: tuple[object, ...] = ('X',)  # type=Optional[List[str]]; help=parameter of sample_model; run_inference.py:514
  laser_run_inference_sample_model__fs_sequence_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference.py:513
  laser_run_inference_sample_model__gly_budget: int = 0  # type=Optional[int]; help=parameter of sample_model; run_inference.py:515
  laser_run_inference_sample_model__ignore_chain_mask_zeros: bool = False  # type=bool; help=parameter of sample_model; run_inference.py:514
  laser_run_inference_sample_model__model: object | None = None  # type=LASErMPNN; help=parameter of sample_model; run_inference.py:512; UNRESOLVED required parameter; no upstream default
  laser_run_inference_sample_model__params: object | None = None  # type=dict; help=parameter of sample_model; run_inference.py:512; UNRESOLVED required parameter; no upstream default
  laser_run_inference_sample_model__repack_all: bool = False  # type=bool; help=parameter of sample_model; run_inference.py:514
  laser_run_inference_sample_model__seq_min_p: float = 0.0  # type=float; help=parameter of sample_model; run_inference.py:513
  laser_run_inference_sample_model__sequence_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference.py:512; UNRESOLVED required parameter; no upstream default
  laser_run_inference_sample_model__use_edo: bool = False  # type=bool; help=parameter of sample_model; run_inference.py:512
  laser_run_inference_sample_model__verbose: bool = False  # type=bool; help=parameter of sample_model; run_inference.py:513

@dataclass(frozen=True)
class LaserRunInferenceTiedKnobs:
  """run_inference_tied.py."""

  laser_run_inference_tied__ala_budget: int = 4  # type=int; help=Max number of alanine residues that can be sampled in exposed non-secondary structured residues when --constrain_ala_gly_sampling_to_exposed_non_secondary_structure is set.; run_inference_tied.py:894
  laser_run_inference_tied__backbone_noise: str = ''  # type=str; help=Inference backbone noise.; run_inference_tied.py:882
  laser_run_inference_tied__budget_residue_sele_string: str = ''  # type=str; help=A ProDy selection string to specify residues to constrain sampling for. If set, only residues that satisfy the selection criteria will be constrained. Ex: "same residue as (chain A and resnum 50)". See ProDy documentation for selection s...; run_inference_tied.py:892
  laser_run_inference_tied__constrain_ala_gly_sampling_to_exposed_non_secondary_structure: bool = False  # type=bool; help=If set, constrains number of ALA and GLY residues that can be sampled in exposed non-secondary structured residues. The max number of ALA and GLY residues is set by the --ala_budget and --gly_budget arguments.; run_inference_tied.py:893
  laser_run_inference_tied__designs_per_input: int = 1  # type=int; help=Number of sequences to design per input structure. Default is 1.; run_inference_tied.py:876
  laser_run_inference_tied__device: str = 'cpu'  # type=str; help=Pytorch style device string. Ex: "cuda:0" or "cpu".; run_inference_tied.py:883
  laser_run_inference_tied__disable_charged_fs: bool = False  # type=bool; help=Disable sampling D,K,R,E residues in the first shell around the ligand.; run_inference_tied.py:899
  laser_run_inference_tied__disabled_residues: str = 'X,C'  # type=str; help=Comma separated list of one letter amino acid codes to disable sampling of. Default is "X,C" which corresponds to any residues with unknown amino acids and Cysteines.; run_inference_tied.py:884
  laser_run_inference_tied__entropy_decoder: bool = False  # type=bool; help=Uses entropy based decoding order. Decodes all residues and selects the lowest entropy residue as next to decode, then recomputes all remaining residues. Takes longer than normal decoding.; run_inference_tied.py:888
  laser_run_inference_tied__fix_beta: bool = False  # type=bool; help=Residues with B-Factor of 1.0 have sequence and rotamer fixed, residues with B-Factor of 0.0 are designed.; run_inference_tied.py:886
  laser_run_inference_tied__fs_calc_burial_hull_alpha_value: float = 9.0  # type=float; help=Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).; run_inference_tied.py:897
  laser_run_inference_tied__fs_calc_ca_distance: float = 10.0  # type=float; help=Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.; run_inference_tied.py:896
  laser_run_inference_tied__fs_no_calc_burial: bool = False  # type=bool; help=Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance; run_inference_tied.py:898
  laser_run_inference_tied__fs_sequence_temp: float | None = None  # type=float; help=Residues around the ligand will be sampled at this temperature, otherwise they default to sequence_temp.; run_inference_tied.py:881
  laser_run_inference_tied__gly_budget: int = 0  # type=int; help=Max number of glycine residues that can be sampled in exposed non-secondary structured residues when --constrain_ala_gly_sampling_to_exposed_non_secondary_structure is set.; run_inference_tied.py:895
  laser_run_inference_tied__ignore_ligand: bool = False  # type=bool; help=Ignore ligands in the input PDB file.; run_inference_tied.py:890
  laser_run_inference_tied__input_pdb_code_1: str | None = None  # type=str; help=Path to the input PDB file.; run_inference_tied.py:873; UNRESOLVED no default literal
  laser_run_inference_tied__input_pdb_code_2: str | None = None  # type=str; help=Path to the input PDB file.; run_inference_tied.py:874; UNRESOLVED no default literal
  laser_run_inference_tied__model_weights: str | None = None  # type=str; help=f'Path to dictionary of torch.save()ed model state_dict and training parameters. Default: {default_weights_path}'; run_inference_tied.py:877; UNRESOLVED default is not a literal (default_weights_path)
  laser_run_inference_tied__noncanonical_aa_ligand: bool = False  # type=bool; help=Featurize a noncanonical amino acid as a ligand.; run_inference_tied.py:891
  laser_run_inference_tied__output_path: str = 'laser_output.pdb'  # type=str; help=Path to the output PDB file.; run_inference_tied.py:879
  laser_run_inference_tied__repack_only: bool = False  # type=bool; help=Only repack residues, do not design new ones.; run_inference_tied.py:889
  laser_run_inference_tied__sequence_temp: str = ''  # type=str; help=Sequence sample temperature.; run_inference_tied.py:880
  laser_run_inference_tied__strict_load: bool = True  # type=bool; help=Small state_dict mismatches are ignored. Don't use this unless any missing parameters aren't learned during training.; run_inference_tied.py:887
  laser_run_inference_tied__tied_probability_interpolation_lambda: float = 0.0  # type=float; help=Lambda value for interpolating between the two model output distributions during tied decoding. Final probabilities are computed as: lambda * prob1 + (1-lambda) * prob2. Default is 0.0 which corresponds to using only the first input stru...; run_inference_tied.py:875

@dataclass(frozen=True)
class LaserRunInferenceTiedSampleModelKnobs:
  """run_inference_tied.py:565:sample_model."""

  laser_run_inference_tied_sample_model__ala_budget: int = 4  # type=Optional[int]; help=parameter of sample_model; run_inference_tied.py:569
  laser_run_inference_tied_sample_model__batch_data1: object | None = None  # type=BatchData; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__batch_data2: object | None = None  # type=BatchData; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__bb_noise: float | None = None  # type=float; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__budget_residue_mask: object | None = None  # type=Optional[torch.Tensor]; help=parameter of sample_model; run_inference_tied.py:569
  laser_run_inference_tied_sample_model__chi_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference_tied.py:566
  laser_run_inference_tied_sample_model__disable_charged_fs: bool = False  # type=bool; help=parameter of sample_model; run_inference_tied.py:570
  laser_run_inference_tied_sample_model__disable_pbar: bool = False  # type=bool; help=parameter of sample_model; run_inference_tied.py:567
  laser_run_inference_tied_sample_model__disabled_residues: tuple[object, ...] = ('X', 'C')  # type=Optional[List[str]]; help=parameter of sample_model; run_inference_tied.py:568
  laser_run_inference_tied_sample_model__fs_sequence_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference_tied.py:567
  laser_run_inference_tied_sample_model__gly_budget: int = 0  # type=Optional[int]; help=parameter of sample_model; run_inference_tied.py:569
  laser_run_inference_tied_sample_model__ignore_chain_mask_zeros: bool = False  # type=bool; help=parameter of sample_model; run_inference_tied.py:568
  laser_run_inference_tied_sample_model__interpolation_lambda: float | None = None  # type=float; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__model: object | None = None  # type=LASErMPNN; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__params: object | None = None  # type=dict; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__repack_all: bool = False  # type=bool; help=parameter of sample_model; run_inference_tied.py:568
  laser_run_inference_tied_sample_model__sequence_temp: object | None = None  # type=Optional[float]; help=parameter of sample_model; run_inference_tied.py:566; UNRESOLVED required parameter; no upstream default
  laser_run_inference_tied_sample_model__use_edo: bool = False  # type=bool; help=parameter of sample_model; run_inference_tied.py:566
  laser_run_inference_tied_sample_model__verbose: bool = False  # type=bool; help=parameter of sample_model; run_inference_tied.py:567

@dataclass(frozen=True)
class LaserRunPredictPartialChargesKnobs:
  """run_predict_partial_charges.py."""

  laser_run_predict_partial_charges__input_pdb: str | None = None  # type=str; help=Path to input PDB file containing ligand coordinates.; run_predict_partial_charges.py:74; UNRESOLVED no default literal
  laser_run_predict_partial_charges__model_weights: str | None = None  # type=str; help=Path to ligand encoder weights.; run_predict_partial_charges.py:76; UNRESOLVED default is not a literal (default_weights_path)
  laser_run_predict_partial_charges__output_pdb: str | None = None  # type=str; help=Output PDB path with predicted partial charges as b-factors.; run_predict_partial_charges.py:75; UNRESOLVED no default literal

@dataclass(frozen=True)
class LaserRunProofreadingKnobs:
  """run_proofreading.py."""

  laser_run_proofreading__device: str = 'cuda:0'  # type=str; help=Device to run the model on.; run_proofreading.py:228
  laser_run_proofreading__disable_inference_dropout: bool = False  # type=bool; help=Disable inference dropout.; run_proofreading.py:230
  laser_run_proofreading__n_decoding_orders: int = 10  # type=int; help=Number of decoding orders to use.; run_proofreading.py:232
  laser_run_proofreading__n_dropouts: int = 10  # type=int; help=Number of dropouts to use.; run_proofreading.py:233
  laser_run_proofreading__output_dir: str | None = None  # type=str; help=Path to an output directory.; run_proofreading.py:227; UNRESOLVED no default literal
  laser_run_proofreading__pdb_file: str | None = None  # type=str; help=Path to the pdb file of the protein complex.; run_proofreading.py:226; UNRESOLVED no default literal
  laser_run_proofreading__repack_all: bool = False  # type=bool; help=Repack all residues.; run_proofreading.py:234
  laser_run_proofreading__selection_string: str = ''  # type=str; help=A Prody selection string to override the default behavior of proofreading the binding site only.; run_proofreading.py:235
  laser_run_proofreading__silent: bool = False  # type=bool; help=Disable printing.; run_proofreading.py:231
  laser_run_proofreading__weights: str | None = None  # type=str; help=f'Path to the weights file. Defaults to {default_weights_path}.'; run_proofreading.py:229; UNRESOLVED default is not a literal (default_weights_path)

@dataclass(frozen=True)
class LaserThreadAndScoreSequencesKnobs:
  """scripts/thread_and_score_sequences.py."""

  laser_thread_and_score_sequences__backbone: str | None = None  # type=str; help=Path to the pdb file of the backbone structure to thread the sequences onto.; scripts/thread_and_score_sequences.py:42; UNRESOLVED no default literal
  laser_thread_and_score_sequences__fasta: str | None = None  # type=str; help=Path to the fasta file of sequences to be threaded onto the input backbone.; scripts/thread_and_score_sequences.py:41; UNRESOLVED no default literal
  laser_thread_and_score_sequences__output_dir: str | None = None  # type=str; help=Path to the output directory where the threaded pdb files will be saved.; scripts/thread_and_score_sequences.py:43; UNRESOLVED no default literal
  laser_thread_and_score_sequences__weights: str | None = None  # type=str; help=Path to the weights file for LASErMPNN. If not provided, default weights will be used.; scripts/thread_and_score_sequences.py:44; UNRESOLVED default is not a literal (default_weights)

@dataclass(frozen=True)
class PottsAssignFixedChainsKnobs:
  """helper_scripts/assign_fixed_chains.py."""

  potts_assign_fixed_chains__chain_list: str = ''  # type=str; help=List of the chains that need to be designed; helper_scripts/assign_fixed_chains.py:32
  potts_assign_fixed_chains__input_path: str | None = None  # type=str; help=Path to the parsed PDBs; helper_scripts/assign_fixed_chains.py:30
  potts_assign_fixed_chains__output_path: str | None = None  # type=str; help=Path to the output dictionary; helper_scripts/assign_fixed_chains.py:31

@dataclass(frozen=True)
class PottsEnergyPredictionKnobs:
  """energy_prediction.py."""

  potts_energy_prediction__config: str | None = None  # type=str; help=; energy_prediction.py:125; UNRESOLVED required; no upstream default

@dataclass(frozen=True)
class PottsMakeBiasAAKnobs:
  """helper_scripts/make_bias_AA.py."""

  potts_make_bias_AA__AA_list: str = ''  # type=str; help=List of AAs to be biased; helper_scripts/make_bias_AA.py:20
  potts_make_bias_AA__bias_list: str = ''  # type=str; help=AA bias strengths; helper_scripts/make_bias_AA.py:21
  potts_make_bias_AA__output_path: str | None = None  # type=str; help=Path to the output dictionary; helper_scripts/make_bias_AA.py:19

@dataclass(frozen=True)
class PottsMakeBiasPerResDictKnobs:
  """helper_scripts/make_bias_per_res_dict.py."""

  potts_make_bias_per_res_dict__input_path: str | None = None  # type=str; help=Path to the parsed PDBs; helper_scripts/make_bias_per_res_dict.py:49
  potts_make_bias_per_res_dict__output_path: str | None = None  # type=str; help=Path to the output dictionary; helper_scripts/make_bias_per_res_dict.py:50

@dataclass(frozen=True)
class PottsMakeFixedPositionsDictKnobs:
  """helper_scripts/make_fixed_positions_dict.py."""

  potts_make_fixed_positions_dict__chain_list: str = ''  # type=str; help=List of the chains that need to be fixed; helper_scripts/make_fixed_positions_dict.py:53
  potts_make_fixed_positions_dict__input_path: str | None = None  # type=str; help=Path to the parsed PDBs; helper_scripts/make_fixed_positions_dict.py:51
  potts_make_fixed_positions_dict__output_path: str | None = None  # type=str; help=Path to the output dictionary; helper_scripts/make_fixed_positions_dict.py:52
  potts_make_fixed_positions_dict__position_list: str = ''  # type=str; help=Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain; helper_scripts/make_fixed_positions_dict.py:54
  potts_make_fixed_positions_dict__specify_non_fixed: bool = False  # type=bool; help=Allows specifying just residues that need to be designed (default: false); helper_scripts/make_fixed_positions_dict.py:55

@dataclass(frozen=True)
class PottsMakePosNegTiedPositionsDictKnobs:
  """helper_scripts/make_pos_neg_tied_positions_dict.py."""

  potts_make_pos_neg_tied_positions_dict__chain_list: str = ''  # type=str; help=List of the chains that need to be fixed; helper_scripts/make_pos_neg_tied_positions_dict.py:61
  potts_make_pos_neg_tied_positions_dict__homooligomer: int = 0  # type=int; help=If 0 do not use, if 1 then design homooligomer; helper_scripts/make_pos_neg_tied_positions_dict.py:63
  potts_make_pos_neg_tied_positions_dict__input_path: str | None = None  # type=str; help=Path to the parsed PDBs; helper_scripts/make_pos_neg_tied_positions_dict.py:59
  potts_make_pos_neg_tied_positions_dict__output_path: str | None = None  # type=str; help=Path to the output dictionary; helper_scripts/make_pos_neg_tied_positions_dict.py:60
  potts_make_pos_neg_tied_positions_dict__pos_neg_chain_betas: str = ''  # type=str; help=Chain beta list for the chain lists provided; 1.0 for the positive design, -0.1 or -0.5 for negative, 0.0 means do not use that chain info; helper_scripts/make_pos_neg_tied_positions_dict.py:65
  potts_make_pos_neg_tied_positions_dict__pos_neg_chain_list: str = ''  # type=str; help=Chain lists to be tied together; helper_scripts/make_pos_neg_tied_positions_dict.py:64
  potts_make_pos_neg_tied_positions_dict__position_list: str = ''  # type=str; help=Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain; helper_scripts/make_pos_neg_tied_positions_dict.py:62

@dataclass(frozen=True)
class PottsMakePssmInputDictKnobs:
  """helper_scripts/make_pssm_input_dict.py."""

  potts_make_pssm_input_dict__PSSM_input_path: str | None = None  # type=str; help=Path to PSSMs saved as npz files.; helper_scripts/make_pssm_input_dict.py:31
  potts_make_pssm_input_dict__jsonl_input_path: str | None = None  # type=str; help=Path where to load .jsonl dictionary of parsed pdbs.; helper_scripts/make_pssm_input_dict.py:32
  potts_make_pssm_input_dict__output_path: str | None = None  # type=str; help=Path where to save .jsonl dictionary with PSSM bias.; helper_scripts/make_pssm_input_dict.py:33

@dataclass(frozen=True)
class PottsMakeTiedPositionsDictKnobs:
  """helper_scripts/make_tied_positions_dict.py."""

  potts_make_tied_positions_dict__chain_list: str = ''  # type=str; help=List of the chains that need to be fixed; helper_scripts/make_tied_positions_dict.py:51
  potts_make_tied_positions_dict__homooligomer: int = 0  # type=int; help=If 0 do not use, if 1 then design homooligomer; helper_scripts/make_tied_positions_dict.py:53
  potts_make_tied_positions_dict__input_path: str | None = None  # type=str; help=Path to the parsed PDBs; helper_scripts/make_tied_positions_dict.py:49
  potts_make_tied_positions_dict__output_path: str | None = None  # type=str; help=Path to the output dictionary; helper_scripts/make_tied_positions_dict.py:50
  potts_make_tied_positions_dict__position_list: str = ''  # type=str; help=Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain; helper_scripts/make_tied_positions_dict.py:52

@dataclass(frozen=True)
class PottsMutationSearchKnobs:
  """mutation_search.py."""

  potts_mutation_search__allowed_from_aas: str | None = None  # type=str; help=String of allowed source AAs; mutation_search.py:776
  potts_mutation_search__allowed_to_aas: str | None = None  # type=str; help=String of allowed target AAs; mutation_search.py:777
  potts_mutation_search__binding_energy_cutoff: float | None = None  # type=float; help=Contact cutoff in Angstroms; mutation_search.py:781
  potts_mutation_search__binding_energy_json: str | None = None  # type=str; help=Path to binding energy JSON; mutation_search.py:780
  potts_mutation_search__cfg_path: str | None = None  # type=str; help=Path to model config YAML; mutation_search.py:764; UNRESOLVED required; no upstream default
  potts_mutation_search__disallowed_chains: tuple[object, ...] = ()  # type=unknown; help=Chains to exclude from mutation; mutation_search.py:775
  potts_mutation_search__energy_mode: str = 'both'  # type=str; choices=('stability', 'binding', 'both'); help=Scoring mode; mutation_search.py:782
  potts_mutation_search__max_keep_per_depth: int = 2000  # type=int; help=Hard cap on kept candidates; mutation_search.py:771
  potts_mutation_search__max_mutations: int = 4  # type=int; help=Max mutation depth; mutation_search.py:768
  potts_mutation_search__no_pareto_front: bool = False  # type=bool; help=Disable Pareto front calculation; mutation_search.py:784
  potts_mutation_search__pdb_paths: object | None = None  # type=unknown; help=List of PDB file paths; mutation_search.py:763; UNRESOLVED required; no upstream default
  potts_mutation_search__per_position_quota: int | None = None  # type=int; help=Max candidates per position; mutation_search.py:772
  potts_mutation_search__plot_dir: str | None = None  # type=str; help=Output directory for plots and CSVs; mutation_search.py:765; UNRESOLVED required; no upstream default
  potts_mutation_search__rrf_k: int = 60  # type=int; help=RRF constant for 'both' mode; mutation_search.py:783
  potts_mutation_search__top_percent: float = 10.0  # type=float; help=Top percentage to keep; mutation_search.py:769
  potts_mutation_search__top_percent_decay_base: float = 1.0  # type=float; help=Decay base for top_percent; mutation_search.py:770

@dataclass(frozen=True)
class PottsParseMultipleChainsKnobs:
  """helper_scripts/parse_multiple_chains.py."""

  potts_parse_multiple_chains__ca_only: bool = False  # type=bool; help=parse a backbone-only structure (default: false); helper_scripts/parse_multiple_chains.py:160
  potts_parse_multiple_chains__input_path: str | None = None  # type=str; help=Path to a folder with pdb files, e.g. /home/my_pdbs/; helper_scripts/parse_multiple_chains.py:158
  potts_parse_multiple_chains__output_path: str | None = None  # type=str; help=Path where to save .jsonl dictionary of parsed pdbs; helper_scripts/parse_multiple_chains.py:159

@dataclass(frozen=True)
class PottsPottsMpnnUtilsPottsMPNNSampleKnobs:
  """potts_mpnn_utils.py:1321:sample."""

  potts_potts_mpnn_utils_PottsMPNN_sample__S_true: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_sample__X: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_sample__bias_AAs_np: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__bias_by_res: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__chain_M_pos: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__chain_encoding_all: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_sample__chain_mask: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_sample__mask: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__omit_AA_mask: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__omit_AAs_np: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__pssm_bias: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__pssm_bias_flag: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__pssm_coef: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__pssm_log_odds_flag: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__pssm_log_odds_mask: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__pssm_multi: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321
  potts_potts_mpnn_utils_PottsMPNN_sample__randn: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_sample__residue_idx: object | None = None  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_sample__temperature: float = 1.0  # type=unknown; help=parameter of sample; potts_mpnn_utils.py:1321

@dataclass(frozen=True)
class PottsPottsMpnnUtilsPottsMPNNTiedSampleKnobs:
  """potts_mpnn_utils.py:1490:tied_sample."""

  potts_potts_mpnn_utils_PottsMPNN_tied_sample__S_true: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__X: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__bias_AAs_np: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__bias_by_res: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__chain_M_pos: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__chain_encoding_all: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__chain_mask: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__mask: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__omit_AA_mask: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__omit_AAs_np: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__pssm_bias: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__pssm_bias_flag: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__pssm_coef: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__pssm_log_odds_flag: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__pssm_log_odds_mask: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__pssm_multi: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__randn: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__residue_idx: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490; UNRESOLVED required parameter; no upstream default
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__temperature: float = 1.0  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__tied_beta: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490
  potts_potts_mpnn_utils_PottsMPNN_tied_sample__tied_pos: object | None = None  # type=unknown; help=parameter of tied_sample; potts_mpnn_utils.py:1490

@dataclass(frozen=True)
class PottsSampleSeqsKnobs:
  """sample_seqs.py."""

  potts_sample_seqs__config: str | None = None  # type=str; help=; sample_seqs.py:414; UNRESOLVED required; no upstream default

@dataclass(frozen=True)
class PottsTrainingKnobs:
  """training/training.py."""

  potts_training__backbone_noise: float = 0.2  # type=float; help=amount of noise added to backbone during training; training/training.py:244
  potts_training__batch_size: int = 10000  # type=int; help=number of tokens for one batch; training/training.py:237
  potts_training__debug: bool = False  # type=bool; help=minimal data loading for debugging; training/training.py:246
  potts_training__dropout: float = 0.1  # type=float; help=dropout level; 0.0 means no dropout; training/training.py:243
  potts_training__gradient_norm: float = -1.0  # type=float; help=clip gradient norm, set to negative to omit clipping; training/training.py:247
  potts_training__hidden_dim: int = 128  # type=int; help=hidden model dimension; training/training.py:239
  potts_training__max_protein_length: int = 10000  # type=int; help=maximum length of the protein complext; training/training.py:238
  potts_training__mixed_precision: bool = True  # type=bool; help=train with mixed precision; training/training.py:248
  potts_training__num_decoder_layers: int = 3  # type=int; help=number of decoder layers; training/training.py:241
  potts_training__num_encoder_layers: int = 3  # type=int; help=number of encoder layers; training/training.py:240
  potts_training__num_epochs: int = 200  # type=int; help=number of epochs to train for; training/training.py:233
  potts_training__num_examples_per_epoch: int = 1000000  # type=int; help=number of training example to load for one epoch; training/training.py:236
  potts_training__num_neighbors: int = 48  # type=int; help=number of neighbors for the sparse graph; training/training.py:242
  potts_training__path_for_outputs: str = './exp_020'  # type=str; help=path for logs and model weights; training/training.py:231
  potts_training__path_for_training_data: str = 'my_path/pdb_2021aug02'  # type=str; help=path for loading training data; training/training.py:230
  potts_training__previous_checkpoint: str = ''  # type=str; help=path for previous model weights, e.g. file.pt; training/training.py:232
  potts_training__reload_data_every_n_epochs: int = 2  # type=int; help=reload training data every n epochs; training/training.py:235
  potts_training__rescut: float = 3.5  # type=float; help=PDB resolution cutoff; training/training.py:245
  potts_training__save_model_every_n_epochs: int = 10  # type=int; help=save model weights every n epochs; training/training.py:234

@dataclass(frozen=True)
class PottsmpnnCfgKnobs:
  """cfg.model/cfg.inference ∪ example YAML."""

  pottsmpnn_cfg__chain_dict_json: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_sample_seqs.yaml:6
  pottsmpnn_cfg__dev: str = 'cuda'  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:1
  pottsmpnn_cfg__inference__bias_AA_json: str = ''  # type=yaml; help=yaml+ast; first read run_utils.py:581; inputs/example_config_sample_seqs.yaml:33
  pottsmpnn_cfg__inference__bias_by_res_json: str = ''  # type=yaml; help=yaml+ast; first read run_utils.py:598; inputs/example_config_sample_seqs.yaml:35
  pottsmpnn_cfg__inference__binding_energy_cutoff: int = 8  # type=yaml; help=yaml+ast; first read sample_seqs.py:308; inputs/example_config_energy_prediction.yaml:22
  pottsmpnn_cfg__inference__binding_energy_json: object | None = None  # type=yaml; help=yaml+ast; first read run_utils.py:649; inputs/example_config_energy_prediction.yaml:21
  pottsmpnn_cfg__inference__binding_energy_optimization: str = 'none'  # type=yaml; help=yaml+ast; first read sample_seqs.py:162; inputs/example_config_sample_seqs.yaml:24
  pottsmpnn_cfg__inference__chain_dict: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:25
  pottsmpnn_cfg__inference__chain_ranges: str = 'inputs/example_chain_ranges.json'  # type=yaml; help=yaml+ast; first read energy_prediction.py:101; inputs/example_config_energy_prediction.yaml:26
  pottsmpnn_cfg__inference__ddG: bool = True  # type=yaml; help=yaml+ast; first read energy_prediction.py:59; inputs/example_config_energy_prediction.yaml:17
  pottsmpnn_cfg__inference__decoding_order_offset: int = 0  # type=yaml; help=yaml+ast; first read sample_seqs.py:202; inputs/example_config_sample_seqs.yaml:21
  pottsmpnn_cfg__inference__exclude_chains: object | None = None  # type=cfg-attr; help=cfg attribute read; run_utils.py:755; UNRESOLVED cfg attribute is read but has no example-YAML default
  pottsmpnn_cfg__inference__filter: bool = False  # type=yaml; help=yaml+ast; first read mutation_search.py:397; inputs/example_config_energy_prediction.yaml:20
  pottsmpnn_cfg__inference__fix_decoding_order: bool = True  # type=yaml; help=yaml+ast; first read sample_seqs.py:201; inputs/example_config_sample_seqs.yaml:20
  pottsmpnn_cfg__inference__fixed_positions_json: str = ''  # type=yaml; help=yaml+ast; first read run_utils.py:554; inputs/example_config_sample_seqs.yaml:30
  pottsmpnn_cfg__inference__max_tokens: int = 20000  # type=yaml; help=yaml+ast; first read run_utils.py:878; inputs/example_config_energy_prediction.yaml:19
  pottsmpnn_cfg__inference__mean_norm: bool = False  # type=yaml; help=yaml+ast; first read mutation_search.py:396; inputs/example_config_energy_prediction.yaml:18
  pottsmpnn_cfg__inference__noise: float = 0.0  # type=yaml; help=yaml+ast; first read energy_prediction.py:34; inputs/example_config_energy_prediction.yaml:24
  pottsmpnn_cfg__inference__num_samples: int = 1  # type=yaml; help=yaml+ast; first read sample_seqs.py:40; inputs/example_config_sample_seqs.yaml:16
  pottsmpnn_cfg__inference__omit_AA_json: str = ''  # type=yaml; help=yaml+ast; first read run_utils.py:572; inputs/example_config_sample_seqs.yaml:32
  pottsmpnn_cfg__inference__omit_AAs: tuple[object, ...] = ()  # type=yaml; help=yaml+ast; first read run_utils.py:607; inputs/example_config_sample_seqs.yaml:36
  pottsmpnn_cfg__inference__optimization_mode: str = 'potts'  # type=yaml; help=yaml+ast; first read sample_seqs.py:130; inputs/example_config_sample_seqs.yaml:22
  pottsmpnn_cfg__inference__optimization_temperature: float = 0.0  # type=yaml; help=yaml+ast; first read sample_seqs.py:39; inputs/example_config_sample_seqs.yaml:23
  pottsmpnn_cfg__inference__optimize_fasta: str = ''  # type=yaml; help=yaml+ast; first read sample_seqs.py:40; inputs/example_config_sample_seqs.yaml:28
  pottsmpnn_cfg__inference__optimize_pdb: bool = False  # type=yaml; help=yaml+ast; first read sample_seqs.py:40; inputs/example_config_sample_seqs.yaml:27
  pottsmpnn_cfg__inference__pssm_bias_flag: bool = False  # type=yaml; help=yaml+ast; first read sample_seqs.py:348; inputs/example_config_sample_seqs.yaml:40
  pottsmpnn_cfg__inference__pssm_json: str = ''  # type=yaml; help=yaml+ast; first read run_utils.py:562; inputs/example_config_sample_seqs.yaml:31
  pottsmpnn_cfg__inference__pssm_log_odds_flag: bool = False  # type=yaml; help=yaml+ast; first read sample_seqs.py:349; inputs/example_config_sample_seqs.yaml:39
  pottsmpnn_cfg__inference__pssm_multi: float = 0.0  # type=yaml; help=yaml+ast; first read sample_seqs.py:348; inputs/example_config_sample_seqs.yaml:38
  pottsmpnn_cfg__inference__pssm_threshold: float = 0.0  # type=yaml; help=yaml+ast; first read sample_seqs.py:198; inputs/example_config_sample_seqs.yaml:37
  pottsmpnn_cfg__inference__skip_gaps: bool = False  # type=yaml; help=yaml+ast; first read energy_prediction.py:62; inputs/example_config_energy_prediction.yaml:23
  pottsmpnn_cfg__inference__temperature: float = 0.1  # type=yaml; help=yaml+ast; first read sample_seqs.py:38; inputs/example_config_sample_seqs.yaml:17
  pottsmpnn_cfg__inference__tied_epistasis: object | None = None  # type=cfg-attr; help=cfg attribute read; sample_seqs.py:361; UNRESOLVED cfg attribute is read but has no example-YAML default
  pottsmpnn_cfg__inference__tied_positions_json: str = ''  # type=yaml; help=yaml+ast; first read run_utils.py:590; inputs/example_config_sample_seqs.yaml:34
  pottsmpnn_cfg__inference__write_pdb: bool = True  # type=yaml; help=yaml+ast; first read sample_seqs.py:124; inputs/example_config_sample_seqs.yaml:29
  pottsmpnn_cfg__input_dir: str = 'inputs/example_pdbs'  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:5
  pottsmpnn_cfg__input_list: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:4; UNRESOLVED YAML defaults disagree: inputs/example_config_energy_prediction.yaml:4='inputs/example_list_energy_prediction.txt'; inputs/example_config_sample_seqs.yaml:4='inputs/example_list_sample_seqs.txt'
  pottsmpnn_cfg__model__check_path: str = 'vanilla_model_weights/pottsmpnn_20.pt'  # type=yaml; help=yaml+ast; first read energy_prediction.py:32; inputs/example_config_energy_prediction.yaml:9
  pottsmpnn_cfg__model__edge_features: int = 128  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:11
  pottsmpnn_cfg__model__hidden_dim: int = 128  # type=yaml; help=yaml+ast; first read energy_prediction.py:33; inputs/example_config_energy_prediction.yaml:10
  pottsmpnn_cfg__model__num_edges: int = 48  # type=yaml; help=yaml+ast; first read energy_prediction.py:34; inputs/example_config_energy_prediction.yaml:14
  pottsmpnn_cfg__model__num_layers: int = 3  # type=yaml; help=yaml+ast; first read energy_prediction.py:34; inputs/example_config_energy_prediction.yaml:13
  pottsmpnn_cfg__model__potts_dim: int = 400  # type=yaml; help=yaml+ast; first read energy_prediction.py:34; inputs/example_config_energy_prediction.yaml:12
  pottsmpnn_cfg__model__vocab: int = 21  # type=yaml; help=yaml+ast; first read energy_prediction.py:29; inputs/example_config_energy_prediction.yaml:15
  pottsmpnn_cfg__mutant_csv: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:7
  pottsmpnn_cfg__mutant_fasta: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:6
  pottsmpnn_cfg__out_dir: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:2; UNRESOLVED YAML defaults disagree: inputs/example_config_energy_prediction.yaml:2='outputs/example_energy_prediction'; inputs/example_config_sample_seqs.yaml:2='outputs/example_sequence_outputs'
  pottsmpnn_cfg__out_name: object | None = None  # type=yaml; help=example YAML key; inputs/example_config_energy_prediction.yaml:3; UNRESOLVED YAML defaults disagree: inputs/example_config_energy_prediction.yaml:3='example_energy_prediction'; inputs/example_config_sample_seqs.yaml:3='sample_run'

# MANUAL (not AST-derivable)
# Input-list rows parsed by PottsMPNN sample_seqs.py:81-94.
# Line format: pdb|designed:chains|fixed:chains
@dataclass(frozen=True)
class pottsmpnn_input_list:
  """Manual input-list fields. The extractor --check diff skips this block."""

  pottsmpnn_input_list__pdb: str | None = None  # type=str; help=PDB stem (text before the first |); sample_seqs.py:81-94
  pottsmpnn_input_list__designed_chains: str | None = None  # type=str; help=designed chains (colon-separated); sample_seqs.py:81-94
  pottsmpnn_input_list__fixed_chains: str | None = None  # type=str; help=fixed chains (colon-separated); sample_seqs.py:81-94
# END MANUAL
