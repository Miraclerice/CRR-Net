## Contents Overview

```shell
├── README.md
├── datasets.py
├── label_info #　Mapping for Voxel Intensity Values ​​and Labels
│   ├── ACDC.txt
│   ├── LPBA40.txt
│   └── Mindboggle.txt
└── trans.py
```

## ROIs Merging Method
The paper reports the mean DSC for each ROI, averaged across all registration pairs, while the box plots illustrate the DSC values for the merged regions. The specific method for combining ROIs is as follows:
### LPBA40

| **Merged ROI**        | **Included Sub-regions**                                                               | **Original Label IDs** |
| --------------------- | -------------------------------------------------------------------------------------- | ---------------------- |
| **Frontal**           | Superior, middle, inferior, middle orbitofrontal, and lateral orbitofrontal gyri (L/R) | 21-26, 29-32           |
| **Parietal**          | Superior parietal gyrus (L/R)                                                          | 43, 44                 |
| **Occipital**         | Superior, middle, and inferior occipital gyri (L/R)                                    | 61-66                  |
| **Temporal**          | Superior, middle, and inferior temporal gyri (L/R)                                     | 81-86                  |
| **Fusiform**          | Fusiform gyrus (L/R)                                                                   | 91, 92                 |
| **Putamen**           | Putamen (L/R)                                                                          | 163, 164               |
| **Hippocampus**       | Hippocampus (L/R)                                                                      | 165, 166               |

### Mindboggle
| **Merged ROI**        | **Included Sub-regions**                                                               | **Original Label IDs** |
| --------------------- | -------------------------------------------------------------------------------------- | ---------------------- |
| **Frontal**           | Caudal middle frontal, lateral orbitofrontal, medial orbitofrontal, rostral middle frontal, superior frontal (L/R) | 1003, 1012, 1014, 1027, 1028, 2003, 2012, 2014, 2027, 2028 |
| **Parietal**          | Inferior parietal, superior parietal (L/R)                                             | 1008, 1029, 2008, 2029                                     |
| **Occipital**         | Lateral occipital (L/R)                                                                | 1011, 2011                                                 |
| **Temporal**          | Inferior temporal, middle temporal, superior temporal, transverse temporal (L/R)       | 1009, 1015, 1030, 1034, 2009, 2015, 2030, 2034             |
| **Fusiform**          | Fusiform (L/R)                                                                         | 1007, 2007                                                 |

For implementation details, please refer to the `Simple_Seg_norm` class in [trans.py](https://github.com/miracledrumstick/CRR-Net/blob/6e60710beefe62ad1c0ac2cad7768d1bc9a59a43/data/trans.py#L43).
