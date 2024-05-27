from debayer import Debayer5x5, Layout

f = Debayer5x5(layout=Layout.RGGB).cpu()


def debayer(output_image):
    bgr_out = f(output_image)
    return bgr_out
