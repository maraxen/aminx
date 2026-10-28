# Third-party model weights

aminx code is MIT-licensed (see `LICENSE`). The model checkpoints below are **converted copies of
third-party weights**, redistributed under the upstream MIT / CC-BY-4.0 licences. Each upstream copyright and
permission notice is reproduced here and must travel with any copy of these weights.

Converted files are the upstream parameters re-serialised as Equinox `.eqx` / `.eqx.zst`; they are not
retrained. Parity of each checkpoint against its upstream is measured (see the parity ledger and
`docs/MODEL_FAMILIES.md`) and is not complete.

| aminx checkpoint id | Upstream | Upstream file |
|---|---|---|
| `pottsmpnn_vanilla_20`, `pottsmpnn_vanilla_30`, `pottsmpnn_sol_20`, `pottsmpnn_sol_30`, `pottsmpnn_ft` | PottsMPNN, <https://github.com/KeatingLab/PottsMPNN> @ `0cb0a58874e373d664114f1735deab77614e4ab5` | `vanilla_model_weights/pottsmpnn_{20,30}.pt`, `soluble_model_weights/sol_pottsmpnn_{20,30}.pt`, `ft_model_weights/potts_ft.pt` |
| `lasermpnn_nothing_heldout`, `lasermpnn_noise_ligandmpnn_split`, `lasermpnn_soluble_65000` | LASErMPNN, <https://github.com/polizzilab/LASErMPNN> @ `e70f2c6d765416f7e29d51bfd6d4e08496438878` | `model_weights/laser_weights_0p1A_nothing_heldout.pt`, `..._noise_ligandmpnn_split.pt`, `soluble_weights_no_heldout_drop_clusters_optstep_65000.pt` |
| `protonpottsmpnn_v6_30` | Proton-PottsMPNN (Jacobsen et al., 2026), oracle pin `09682abf` | `checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt` |
| `abmpnn_v_48_020` | AbMPNN (Dreyer et al., 2023), Zenodo DOI `10.5281/zenodo.8164693` (CC BY 4.0) | `AbMPNN/abmpnn.pt` |
| `antibmpnn_v_48_000`, `antibmpnn_v_48_010`, `antibmpnn_v_48_020`, `antibmpnn_v_48_030` | AntiBMPNN (Zeyu Sun et al., 2024), <https://github.com/zeysun/AntiBMPNN>, Zenodo DOI `10.5281/zenodo.13387792` (CC BY 4.0 / MIT) | `antibmpnn_model_weights/antibmpnn_{000,010,020,030}.pt` |
| `hypermpnn_v_48_002`, `hypermpnn_v_48_010`, `hypermpnn_v_48_020`, `hypermpnn_v_48_030` | HyperMPNN, <https://github.com/meilerlab/HyperMPNN> (MIT) | `retrained_models/v48_{002_epoch240,010_epoch300,020_epoch300,030_epoch300}_hyper.pt` |
| `cyclicmpnn_v_48_010` | CyclicMPNN, <https://github.com/ParisaH-Lab/CyclicMPNN> (MIT) | `cyclicmpnn_weights/cyclicmpnn_48_010.pt` |
| `ca_v_48_002`, `ca_v_48_010`, `ca_v_48_020` | ProteinMPNN CA-only, <https://github.com/dauparas/ProteinMPNN> (MIT) | `ca_model_weights/v_48_{002,010,020}.pt` |
| `thermo_mpnn_v1_default` | ThermoMPNN, <https://github.com/Kuhlman-Lab/ThermoMPNN> (MIT) | `models/thermoMPNN_default.pt` |
| `thermo_mpnn_v2_ens1`, `thermo_mpnn_v2_ens2`, `thermo_mpnn_v2_ens3`, `thermo_mpnn_d_ens1`, `thermo_mpnn_d_ens2`, `thermo_mpnn_d_ens3` | ThermoMPNN-D, <https://github.com/Kuhlman-Lab/ThermoMPNN-D> (MIT) | `model_weights/ThermoMPNN-ens{1,2,3}.ckpt`, `model_weights/ThermoMPNN-D-ens{1,2,3}.ckpt` |
| `caliby`, `soluble_caliby`, `soluble_caliby_v1`, `caliby_distill`, `soluble_caliby_distill`, `caliby_packer_000`, `caliby_packer_010`, `caliby_packer_030` | Caliby (Shuai, Lu, Bhatti, Kouba, Huang, 2025/2026), <https://github.com/ProteinDesignLab/caliby>, Zenodo DOI `10.5281/zenodo.18644985` (Apache-2.0) | `model_params/caliby/*.ckpt` |
| `fampnn_0_0`, `fampnn_0_3`, `fampnn_0_3_cath` | FAMPNN (Shuai, Widatalla, Possu et al., ICML 2025), <https://github.com/richardshuai/fampnn> (MIT) | `weights/fampnn_{0_0,0_3,0_3_cath}.pt` |

The Proton-PottsMPNN licence states that its MIT grant covers the `checkpoints/` directory. Its bundled
`foundry/` directory (IPD rc-foundry, BSD 3-Clause, Copyright (c) 2025 Institute for Protein Design,
University of Washington) is not redistributed here.

If you use these weights in academic work, cite the upstream publications.

---

## PottsMPNN

MIT License

Copyright (c) 2022 Justas Dauparas
Copyright (c) 2026 Foster Birnbaum, Amy E. Keating.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## LASErMPNN

Copyright (c) 2025 Benjamin Fry, Nicholas Polizzi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Proton-PottsMPNN

MIT License

Copyright (c) 2026 Christian P. Jacobsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ProteinMPNN (CA-Only) & CyclicMPNN

MIT License

Copyright (c) 2022 Justas Dauparas

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## AbMPNN

Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2023 Frédéric A. Dreyer, Daniel Cutting, Charlotte M. Deane, Jan Ludwiczak, Aleksandr Karosiene (Zenodo DOI: 10.5281/zenodo.8164693).

## HyperMPNN

MIT License

Copyright (c) 2024 Meiler Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ThermoMPNN

MIT License

Copyright (c) 2023 Kuhlman-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ThermoMPNN-D

MIT License

Copyright (c) 2024 Henry Dieckhaus

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## AntiBMPNN

Creative Commons Attribution 4.0 International (CC BY 4.0) / MIT License

Copyright (c) 2024 Zeyu Sun et al. (Zenodo DOI: 10.5281/zenodo.13387792; <https://github.com/zeysun/AntiBMPNN>).

## Caliby

Apache License, Version 2.0

Copyright (c) 2025 Richard Shuai, Zhaoyang Lu, Talal Bhatti, Petr Kouba, Po-Ssu Huang, Protein Design Lab (<https://github.com/ProteinDesignLab/caliby>; Zenodo DOI: 10.5281/zenodo.18644985).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance
with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

## FAMPNN

MIT License

Copyright (c) 2025 Richard Shuai, Talal Widatalla, Po-Ssu Huang (<https://github.com/richardshuai/fampnn>).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
