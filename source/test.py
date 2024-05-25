import cv2
from torch import nn

from source.data import InMemoryDataset
from source.debayer import debayer

device = "cpu"


class newMO(nn.Module):

    def forward(self, x):
        return x, x


def get_label_fn(path):
    # Заменить папку "INPUT_IMAGES" на "GT_IMAGES"
    gt_path = path.replace("samples", "output")
    # Удалить приписку в конце имени файла, если она существует, и заменить на "_gt" с сохранением формата
    gt_path = gt_path.replace(".CR2", ".png")
    return gt_path


dataset = InMemoryDataset("", "samples/*.*", get_label_fn,
                          return_name=False)

print(len(dataset.rawList))
print(len(InMemoryDataset.rawList))

counter = 0;

model = newMO()
model.eval()
model.cpu()




for sample in dataset:
    counter += 1
    img, raw, gt = sample
    output_image, _ = model(img)
    bgr_out = debayer(output_image, raw)

    cv2.imwrite("test_output/" + str(counter) + ".png", bgr_out,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])



    cv2.imwrite(f"test_output/{counter}_gt.png", gt, [cv2.IMWRITE_PNG_COMPRESSION, 0])
