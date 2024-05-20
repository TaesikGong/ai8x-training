Here is the revised version of your README file for the machine learning project:

---

## Reproducing DEX (NeurIPS '24)

This repository reproduces the accuracy reported in the paper.

The codebase is forked from the [AI8X Training repository](https://github.com/analogdevicesinc/ai8x-training), the official repository for MAX78000 and MAX78002 AI accelerators, and additional implementations are added for the DEX proejct.

### Tested Environment

- Ubuntu 22.04
- NVIDIA A40 GPUs

### Step-by-Step Guide


1. Move to the root directory:
   ```bash
   cd DEX_code
   ```

2. Build the Docker image:
   ```bash
   ./docker/build_image.sh
   ```

3. Run the Docker container:
   ```bash
   ./docker/run_container.sh
   ```

4. Execute the container:
   ```bash
   ./docker/exec_container.sh
   ```

   This starts the SSH service in the Docker container.
 
   **Password**: `root`

5. Run the training script:
   ```bash
   ./scripts/custom/data_folding/train_all.sh
   ```

6. Download and extract datasets:

   - **Caltech101**:
     - Download: [Caltech101](https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp)
     - Extract under `data/Caltech101/caltech101/101_ObjectCategories`
   
   - **Caltech256**:
     - Download: [Caltech256](https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK)
     - Extract under `data/Caltech256/caltech256/256_ObjectCategories/`
   
   - **ImageNette**:
     - Download: [ImageNette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)
     - Extract under `data/Imagenette/`
   
   - **Food101**:
     - The required data will be automatically downloaded.

7. Modify the `train_all.sh` file to run specific experiments as needed. By default, it launches training for ImageNette with SimpleNet

### TODO

- Remove user-specific information (e.g., username `taesik`).
- Set ImageNette as the default dataset in `train_all.sh`.
- Remove unnecessary directories:
   ```bash
   rm -rf ./.git ./data ./notebooks ./logs
   ```
-
   ```bash
   docker exec -it ai8x-training-container sh -c 'su -c "service ssh start" root && cd ~/git/DEX_code && /bin/bash'
   ```

   **Note**: Replace `~/git/ai8x-training` with `~/git/DEX_code` in the command above.

---

This revised version aims to be clear and concise, making it easier for users to follow the setup instructions and understand the necessary steps to run the project.