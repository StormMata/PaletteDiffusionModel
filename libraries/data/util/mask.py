# Copyright (c) OpenMMLab. All rights reserved.
import math

# import cv2
import random
import numpy as np
from PIL import Image, ImageDraw

def bottom_mask_half(img_shape, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    sector_id = np.random.randint(1, 5)

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)

    mask[0:height, 0:25, :] = 1

    return mask

def bottom_mask_4(img_shape, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    sector_id = np.random.randint(1, 5)

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)

    if sector_id == 1:
        mask[0:height, 0:50, :] = 1
    elif sector_id == 2:
        mask[0:height, 50:100, :] = 1
    elif sector_id == 3:
        mask[0:height, 100:150, :] = 1
    elif sector_id == 4:
        mask[0:height, 150:200, :] = 1

    return mask

def bottom_mask_10(img_shape, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    sector_id = np.random.randint(1, 21)

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)

    if sector_id == 1:
        mask[0:height, 0:10, :] = 1
    elif sector_id == 2:
        mask[0:height, 10:20, :] = 1
    elif sector_id == 3:
        mask[0:height, 20:30, :] = 1
    elif sector_id == 4:
        mask[0:height, 30:40, :] = 1
    elif sector_id == 5:
        mask[0:height, 40:50, :] = 1
    elif sector_id == 6:
        mask[0:height, 50:60, :] = 1
    elif sector_id == 7:
        mask[0:height, 60:70, :] = 1
    elif sector_id == 8:
        mask[0:height, 70:80, :] = 1
    elif sector_id == 9:
        mask[0:height, 80:90, :] = 1
    elif sector_id == 10:
        mask[0:height, 90:100, :] = 1
    elif sector_id == 11:
        mask[0:height, 100:110, :] = 1
    elif sector_id == 12:
        mask[0:height, 110:120, :] = 1
    elif sector_id == 13:
        mask[0:height, 120:130, :] = 1
    elif sector_id == 14:
        mask[0:height, 130:140, :] = 1
    elif sector_id == 15:
        mask[0:height, 140:150, :] = 1
    elif sector_id == 16:
        mask[0:height, 150:160, :] = 1
    elif sector_id == 17:
        mask[0:height, 160:170, :] = 1
    elif sector_id == 18:
        mask[0:height, 170:180, :] = 1
    elif sector_id == 19:
        mask[0:height, 180:190, :] = 1
    elif sector_id == 20:
        mask[0:height, 190:200, :] = 1

    return mask

def random_cropping_bbox(img_shape = (256, 256), mask_mode = 'onedirection'):
    h, w = img_shape
    if mask_mode == 'onedirection':
        _type = np.random.randint(0, 4)
        if _type == 0:
            top, left, height, width =    0,    0,    h, w//2
        elif _type == 1:
            top, left, height, width =    0,    0, h//2,    w
        elif _type == 2:
            top, left, height, width = h//2,    0, h//2,    w
        elif _type == 3:
            top, left, height, width =    0, w//2,    h, w//2
    else:
        target_area = (h*w)//2
        width = np.random.randint(target_area//h, w)
        height = target_area//width
        if h==height:
            top = 0
        else:
            top = np.random.randint(0, h-height)
        if w==width:
            left = 0
        else:
            left = np.random.randint(0, w-width)
    return (top, left, height, width)

def random_bbox(img_shape = (96, 200), max_bbox_shape = (48, 100), max_bbox_delta = 10, min_margin = 2):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)
        
    img_h, img_w = img_shape[:2]
    max_mask_h,  max_mask_w   = max_bbox_shape
    max_delta_h, max_delta_w  = max_bbox_delta
    margin_h,    margin_w     = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top  = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top  = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top  = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    # return (top, left, h, w)

    # Create an array with zeros and set the inside of the bounding box to 1
    mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    mask[top:top + h, left:left + w, 0] = 1
    
    return mask

def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[0:height, 0:50, :] = 1

    return mask

def brush_stroke_mask(img_shape,
                      out_channels,
                      num_vertices = (4, 12),
                      mean_angle   = 2 * math.pi / 5,
                      angle_range  = 2 * math.pi / 15,
                      brush_width  = (12, 40),
                      max_loops    = 4,
                      dtype        = 'uint8'):
    """Generate free-form mask.

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
        dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :, None]

    mask = np.repeat(mask, len(out_channels), axis=2)

    # mask = np.transpose(mask, (2, 0, 1))

    # mask = mask * np.array(out_channels).reshape(len(out_channels), 1, 1)

    mask = mask * out_channels

    # print('MAKING MASK')
    # print(mask.shape)

    return mask

# def random_irregular_mask(img_shape,
#                           num_vertices=(4, 8),
#                           max_angle=4,
#                           length_range=(10, 100),
#                           brush_width=(10, 40),
#                           dtype='uint8'):
#     """Generate random irregular masks.

#     This is a modified version of free-form mask implemented in
#     'brush_stroke_mask'.

#     We prefer to use `uint8` as the data type of masks, which may be different
#     from other codes in the community.

#     TODO: Rewrite the implementation of this function.

#     Args:
#         img_shape (tuple[int]): Size of the image.
#         num_vertices (int | tuple[int]): Min and max number of vertices. If
#             only give an integer, we will fix the number of vertices.
#             Default: (4, 8).
#         max_angle (float): Max value of angle at each vertex. Default 4.0.
#         length_range (int | tuple[int]): (min_length, max_length). If only give
#             an integer, we will fix the length of brush. Default: (10, 100).
#         brush_width (int | tuple[int]): (min_width, max_width). If only give
#             an integer, we will fix the width of brush. Default: (10, 40).
#         dtype (str): Indicate the data type of returned masks. Default: 'uint8'

#     Returns:
#         numpy.ndarray: Mask in the shape of (h, w, 1).
#     """

#     h, w = img_shape[:2]

#     mask = np.zeros((h, w), dtype=dtype)
#     if isinstance(length_range, int):
#         min_length, max_length = length_range, length_range + 1
#     elif isinstance(length_range, tuple):
#         min_length, max_length = length_range
#     else:
#         raise TypeError('The type of length_range should be int'
#                         f'or tuple[int], but got type: {length_range}')
#     if isinstance(num_vertices, int):
#         min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
#     elif isinstance(num_vertices, tuple):
#         min_num_vertices, max_num_vertices = num_vertices
#     else:
#         raise TypeError('The type of num_vertices should be int'
#                         f'or tuple[int], but got type: {num_vertices}')

#     if isinstance(brush_width, int):
#         min_brush_width, max_brush_width = brush_width, brush_width + 1
#     elif isinstance(brush_width, tuple):
#         min_brush_width, max_brush_width = brush_width
#     else:
#         raise TypeError('The type of brush_width should be int'
#                         f'or tuple[int], but got type: {brush_width}')

#     num_v = np.random.randint(min_num_vertices, max_num_vertices)

#     for i in range(num_v):
#         start_x = np.random.randint(w)
#         start_y = np.random.randint(h)
#         # from the start point, randomly setlect n \in [1, 6] directions.
#         direction_num = np.random.randint(1, 6)
#         angle_list = np.random.randint(0, max_angle, size=direction_num)
#         length_list = np.random.randint(
#             min_length, max_length, size=direction_num)
#         brush_width_list = np.random.randint(
#             min_brush_width, max_brush_width, size=direction_num)
#         for direct_n in range(direction_num):
#             angle = 0.01 + angle_list[direct_n]
#             if i % 2 == 0:
#                 angle = 2 * math.pi - angle
#             length = length_list[direct_n]
#             brush_w = brush_width_list[direct_n]
#             # compute end point according to the random angle
#             end_x = (start_x + length * np.sin(angle)).astype(np.int32)
#             end_y = (start_y + length * np.cos(angle)).astype(np.int32)

#             cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
#             start_x, start_y = end_x, end_y
#     mask = np.expand_dims(mask, axis=2)

#     return mask


def get_irregular_mask(img_shape, area_ratio_range=(0.15, 0.5), **kwargs):
    """Get irregular mask with the constraints in mask ratio

    Args:
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    mask = random_irregular_mask(img_shape, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) /
                           (img_shape[0] * img_shape[1])) < max_ratio:
        mask = random_irregular_mask(img_shape, **kwargs)

    return mask

def get_custom_mask(img_shape, mask_dir, dtype='uint8'):
    '''
    Return a mask based on a user-specified numpy array
    '''

    ## Read the mask file
    mask = np.load(mask_dir)
    mask = mask.astype(dtype)

    ## Check that the mask dimensions are correct 
    height, width = img_shape[:2]
    assert mask.shape[0] == height
    assert mask.shape[1] == width

    if len(img_shape) == 3:  # Volume data
        depth = img_shape[2]
        assert mask.shape[2] == depth

    return mask

def blob_mask(img_shape, out_channels, dtype='uint8'):
    img_h, img_w = img_shape[:2]
    total_pixels = img_h * img_w
    
    # Set minimum and maximum mask area
    min_area = 0.10 * total_pixels  # 10% of the image size
    max_area = 0.33 * total_pixels  # 33% of the image size
    
    # Create a blank mask
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    
    # Create a random number of vertices for the blob
    num_vertices = np.random.randint(4, 12)
    
    # Generate random vertices
    vertices = []
    for _ in range(num_vertices):
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)
        vertices.append((x, y))
    
    # Draw the irregular polygon on the mask
    draw.polygon(vertices, fill=1)

    # Convert to numpy array
    mask_np = np.array(mask).astype(dtype)
    
    # Calculate the current mask area
    mask_area = np.sum(mask_np)
    
    # Ensure mask area is within the required range
    if mask_area < min_area or mask_area > max_area:
        scale_factor = np.sqrt(min_area / mask_area)
        mask = mask.resize((int(img_w * scale_factor), int(img_h * scale_factor)), resample=Image.BILINEAR)
        mask = mask.resize((img_w, img_h), resample=Image.BILINEAR)
        mask_np = np.array(mask).astype(dtype)
    
    # Add an extra dimension for channels
    mask_np = mask_np[:, :, None]
    
    # Repeat the mask to match the number of channels
    mask_np = np.repeat(mask_np, len(out_channels), axis=2)
    
    # Apply mask only where out_channels is 1
    out_channels = np.array(out_channels).reshape(1, 1, len(out_channels))
    mask_np = mask_np * out_channels

    return mask_np