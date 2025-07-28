## CRR-Net: A Correlation Reconstruction and Refinement Network for Deformable Medical Image Registration
Official Pytorch implementaion of CRR-Net: A Correlation Reconstruction and Refinement Network for Deformable Medical Image Registration.

<p align="center">
  <img src="assets/framework.png" width="800"/>
</p>


### Requirements
The code has been tested on python 3.8 and pytorch 1.12.1.

### Datasets
- LPBA40 [[link](https://resource.loni.usc.edu/resources/atlases-downloads/)]
- Mindboggle [[link](https://osf.io/yhkde/)]
- ACDC [[link](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)]

### Usage
- Train CRR-Net
    ```shell
    sbatch train.sh
    ```

- Test CRR-Net

    ```shell
    sbatch infer.sh
    ```
