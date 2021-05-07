from PIL import Image
import os


def convert_to_eps(image_png, dir):
    im = Image.open(dir + image_png)
    print(im.mode)
    fig = im.convert('RGB')
    fig.save(''.join([dir, image_png[:-4], '.eps']), lossless=True)


def convert_to_pdf(image_png, dir):

    image1 = Image.open(dir + image_png)
    im1 = image1.convert('RGB')
    im1.save(''.join([dir, image_png[:-4], '.pdf']))


def main():
    dir_imgs = '/home/nearlab/Downloads/lumen_segmentation_arXiv/images/'
    list_imgs = [element for element in os.listdir(dir_imgs) if element[-4:] == '.eps']
    for image in list_imgs:
        convert_to_pdf(image, dir_imgs)


if __name__ == "__main__":
    main()