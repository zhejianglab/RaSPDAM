import matplotlib.pyplot as plt
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reager_writer import NaturalImage2DIO

nnUNet_results = 'nnunet_models/nnUNet_trained_models'

if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
        perform_everything_on_gpu=False
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, "Dataset011_FRB/nnUNetTrainer__nnUNetPlans__2d"),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name="checkpoint_final.pth",
    )

    img, props = NaturalImage2DIO().read_images(
        ['218.png']
    )

    print(img.shape)

    ret = predictor.predict_single_npy_array(img, {'spacing': (999, 1, 1)}, None, None, True)

    plt.imshow(ret[0][0])
    plt.savefig("test.png")
