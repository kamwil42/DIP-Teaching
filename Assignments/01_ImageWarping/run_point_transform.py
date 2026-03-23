import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping
    try:
        # For inverse warp compute mapping target to source
        p = np.array(target_pts, dtype=np.float32)
        q = np.array(source_pts, dtype=np.float32)

        # Check if amount of points is correct
        if len(p) == 0 or len(q) == 0:
            return image
        elif len(p) != len(q):
            gr.Warning("Number of source and target points must match.")
            return image

        height, width = image.shape[:2]

        y_coords, x_coords = np.mgrid[0:height, 0:width]
        v = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

        diff = v[:, :, None, :] - p[None, None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        weights = 1.0 / (dist_sq ** alpha + eps)

        sum_w = np.sum(weights, axis=-1, keepdims=True)

        p_star = np.sum(weights[..., None] * p[None, None, :, :], axis=2) / sum_w
        q_star = np.sum(weights[..., None] * q[None, None, :, :], axis=2) / sum_w

        p_hat = p[None, None, :, :] - p_star[:, :, None, :]
        q_hat = q[None, None, :, :] - q_star[:, :, None, :]

        # B = sum w_i * (p_hat * p_hat^T)
        # C = sum w_i * (p_hat * q_hat^T)
        mat_b = np.einsum('hwn, hwni, hwnj->hwij', weights, p_hat, p_hat)
        mat_c = np.einsum('hwn, hwni, hwnj->hwij', weights, p_hat, q_hat)

        # Regularized inverse of B to avoid numerical instability
        mat_b_inv = np.linalg.inv(mat_b + eps * np.eye(2)[None, None, :, :])

        mat = np.matmul(mat_b_inv, mat_c)

        delta = v - p_star

        # Mapped source coordinate for each output pixel
        source_coords = q_star + np.einsum('hwi, hwij->hwj', delta, mat)

        # Split into x and y maps and clamp to valid image bounds
        map_x = np.clip(source_coords[..., 0].astype(np.float32), 0, width - 1)
        map_y = np.clip(source_coords[..., 1].astype(np.float32), 0, height - 1)

        # Final inverse warping
        warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    except Exception as e:
        print("An exception occurred: {}".format(e))

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
