import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config as C

_FRONTAL_TEMPLATE1c = cv2.imread(C.FRONTAL_TEMPLATE_PATH, 0)
_LATERAL_TEMPLATE1c = cv2.imread(C.LATERAL_TEMPLATE_PATH, 0)


def template_match(img, template, tm_method=cv2.TM_CCOEFF_NORMED, init_resize=(256, 256)):
    h, w = template.shape
    is_color = img.ndim > 2
    if is_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1c256 = cv2.resize(img, init_resize, interpolation=cv2.INTER_LINEAR)

    res = cv2.matchTemplate(img1c256, template, tm_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc if tm_method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    (x, y), (x1, y1) = top_left, bottom_right
    if is_color:
        return cv2.cvtColor(img1c256[y:y1, x:x1], cv2.COLOR_GRAY2BGR)
    return img1c256[y:y1, x:x1]


class TemplateCrop(A.ImageOnlyTransform):
    """Match image to template of either a lateral or frontal view, resizing and cropping in the process.

    Args:
        tm_method (int, optional): cv2 template matching method option. Defaults to cv2.TM_CCOEFF_NORMED.
        init_resize (tuple(int,int), optional): Initial image resize dimensions before applying template matching.
            Defaults to (256,256). 
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, tm_method=cv2.TM_CCOEFF_NORMED, init_resize=(256, 256), always_apply=False, p=1.0):
        super(TemplateCrop, self).__init__(always_apply, p)
        self.tm_method = tm_method
        self.init_resize = init_resize

    def apply(self, image, **params):
        template = _LATERAL_TEMPLATE1c if params.get('is_lateral') else _FRONTAL_TEMPLATE1c
        return template_match(image, template, self.tm_method, self.init_resize)

    def get_transform_init_args_names(self):
        return ("tm_method", "init_resize")


def advprop(image, **kwargs):
    return image * 2.0 - 1.0


def get_transforms(varient='train', tfms_lib='albu', imgsize=(244, 244), color=True):
    if tfms_lib == 'albu':
        outpipe=[A.Normalize(),ToTensorV2()] if color else [A.Normalize(mean=0.449,std=0.226)]
        transform = A.Compose([
            A.RandomScale((-0.02, 0.02)),
            A.OneOf([
                TemplateCrop(init_resize=(256, 256), p=1.0),
                A.Compose([A.Resize(256, 256, p=1.0), A.RandomCrop(*imgsize, p=1.0)]),
                # A.RandomCrop(*imgsize, p=1.0)
            ], p=1.0),
            A.CLAHE(p=0.5),
            A.HorizontalFlip(),
            A.Rotate((-7, 7)),  # ,border_mode=cv2.BORDER_CONSTANT),
            A.IAAAffine(shear=(-5, 5)),
            # A.Cutout(8,8,8),
            *outpipe

            # A.Lambda(advprop),

        ])

        tta_augments = A.Compose([
            A.OneOf([
                TemplateCrop(init_resize=(256, 256), p=1.0),
                A.Compose([A.Resize(256, 256, p=1.0), A.CenterCrop(*imgsize, p=1.0)])
            ], p=1.0),
            A.OneOf([
                A.HorizontalFlip(),
                A.Rotate((-7, 7)),  # border_mode=cv2.BORDER_CONSTANT),
                A.IAAAffine(shear=(-5, 5)),
                A.NoOp()
            ], p=1.0),
            *outpipe

        ])

    elif tfms_lib == 'torch':
        from PIL import Image
        import torchvision.transforms as T
        transform = T.Compose([
            # T.RandomCrop(512,8,padding_mode='reflect') ,
            T.CenterCrop(imgsize),
            T.RandomHorizontalFlip(),
            T.RandomRotation(7),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # T.RandomErasing(inplace=True)
        ])
        tta_augments = T.Compose([T.CenterCrop(32), T.ToTensor()])

    return transform if varient == 'train' else tta_augments