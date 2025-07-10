import marimo

__generated_with = "0.11.18"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import imgaug as ia
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage as SegOnImg
    import cv2
    from tqdm import tqdm
    return SegOnImg, cv2, h5py, ia, mo, np, plt, tqdm


@app.cell
def _(mo):
    filebrowser = mo.ui.file_browser(
        #initial_path='/mnt/d/rich/hyper-scope/data/interim/',
        multiple=False,
        filetypes=['.h5']
    )

    filebrowser
    return (filebrowser,)


@app.cell
def _(filebrowser, h5py, mo):
    with h5py.File(filebrowser.value[0].path) as _f:
        slider = mo.ui.slider(start=0, stop=len(_f['images']), step=1, debounce=True)

    slider
    return (slider,)


@app.cell
def _(SegOnImg, filebrowser, h5py, ia, np, plt, slider, tqdm):
    print(slider.value)
    def _(i):
        cells = []
        with h5py.File(filebrowser.value[0].path) as f:
            for i in tqdm(range(i+3, i+6)):
                img = np.repeat((f['images'][i] * 255).astype(np.uint8)[..., np.newaxis], 3, axis=2)
                map = f['inferences'][i]
                print(np.unique(f['inferences'][i]))
                som = SegOnImg(f['inferences'][i], f['images'][i].shape)
                cells.append(img)                                         # column 1
                cells.append(som.draw_on_image(img)[0])                # column 2

        grid_image = ia.draw_grid(cells, cols=2)
        return grid_image
    plt.imshow(_(slider.value))
    plt.show()
    return


if __name__ == "__main__":
    app.run()
