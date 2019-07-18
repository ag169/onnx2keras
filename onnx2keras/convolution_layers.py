import keras.layers
import logging
from .utils import ensure_tf_type, ensure_numpy_type
from math import floor


def convert_conv(node, params, layers, node_name):
    """
    Convert convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:conv')

    if len(node.input) == 3:
        logger.debug('Conv with bias')
        # Has bias
        has_bias = True
        W = ensure_numpy_type(layers[node.input[1]])
        bias = ensure_numpy_type(layers[node.input[2]])

    elif len(node.input) == 2:
        logger.debug('Conv without bias')
        has_bias = False
        W = ensure_numpy_type(layers[node.input[1]])
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]])

    n_groups = params['group'] if 'group' in params else 1

    if len(W.shape) == 5:  # 3D conv
        raise NotImplementedError('Not implemented')
    elif len(W.shape) == 4:  # 2D conv

        dilation = params['dilations'] if 'dilations' in params else [1, 1]
        pads = params['pads'] if 'pads' in params else [0, 0, 0, 0]
        strides = params['strides'] if 'strides' in params else [1, 1]

        ip_shape = input_0._keras_shape
        h_in = ip_shape[2]
        w_in = ip_shape[3]

        logger.debug('2D convolution')

        W = W.transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape
        in_channels = channels_per_group * n_groups

        padding_mode = 'valid'

        if pads[0] > 0 or pads[1] > 0:
            if h_in is not None and w_in is not None:
                h_out = (h_in + strides[0] - 1) // strides[0]
                w_out = (w_in + strides[1] - 1) // strides[1]

                h_out_conv = floor(1.0 + (h_in + (pads[0] + pads[2]) - dilation[0] * (height - 1) - 1.0) / strides[0])
                w_out_conv = floor(1.0 + (w_in + (pads[1] + pads[3]) - dilation[1] * (width - 1) - 1.0) / strides[1])

                if h_out_conv == h_out and w_out_conv == w_out and abs(pads[0] - pads[2]) <= 1 \
                        and abs(pads[1] - pads[3]) <= 1:
                    logger.debug("Paddings exist and are corresponding to 'same', setting mode to 'same'")
                    padding_mode = 'same'
            else:
                h_diff = floor(1.0 + (pads[0] + pads[2] - dilation[0] * (height - 1) - 1) / strides[0])
                w_diff = floor(1.0 + (pads[1] + pads[3] - dilation[1] * (height - 1) - 1) / strides[1])

                if h_diff == 0 and w_diff == 0 and abs(pads[0] - pads[2]) <= 1 and abs(pads[1] - pads[3]) <= 1:
                    logger.debug("Paddings exist and are corresponding to 'same', setting mode to 'same'")
                    padding_mode = 'same'

            if padding_mode == 'valid':
                logger.debug("Paddings exist and are not corresponding to 'same', add ZeroPadding layer")
                padding_name = node_name + '_pad'
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=(pads[0], pads[1]),
                    name=padding_name
                )
                layers[padding_name] = input_0 = padding_layer(input_0)

        if n_groups == in_channels and n_groups != 1:
            logger.debug('Number of groups is equal to input channels, use DepthWise convolution')
            W = W.transpose(0, 1, 3, 2)
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.DepthwiseConv2D(
                kernel_size=(height, width),
                strides=(strides[0],strides[1]),
                padding=padding_mode,
                use_bias=has_bias,
                activation=None,
                depth_multiplier=1,
                weights=weights,
                dilation_rate=dilation,
                bias_initializer='zeros', kernel_initializer='zeros',
                name=node_name
            )
            layers[node_name] = conv(input_0)

        elif n_groups != 1:
            logger.debug('Number of groups more than 1, but less than number of in_channel, use group convolution')

            # Example from https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
            def target_layer(x, groups=n_groups, stride_y=strides[0], stride_x=strides[1]):
                import tensorflow as tf
                x = tf.transpose(x, [0, 2, 3, 1])

                def convolve_lambda(i, k):
                    import tensorflow as tf
                    return tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding_mode)

                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W.transpose(0, 1, 2, 3))
                output_groups = [convolve_lambda(i, k) for i, k in zip(input_groups, weight_groups)]

                layer = tf.concat(axis=3, values=output_groups)

                layer = tf.transpose(layer, [0, 3, 1, 2])
                return layer

            lambda_layer = keras.layers.Lambda(target_layer)
            layers[node_name] = lambda_layer(input_0)

        else:
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=(height, width),
                strides=(strides[0], strides[1]),
                padding=padding_mode,
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation,
                bias_initializer='zeros', kernel_initializer='zeros',
                name=node_name
            )
            layers[node_name] = conv(input_0)

    else:  # 1D conv
        raise NotImplementedError('Not implemented')


def convert_convtranspose(node, params, layers, node_name):
    """
    Convert transposed convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:convtranpose')

    if len(node.input) == 3:
        logger.debug('ConvTranspose with bias')
        # Has bias
        has_bias = True
        W = ensure_numpy_type(layers[node.input[1]])
        bias = ensure_numpy_type(layers[node.input[2]])

    elif len(node.input) == 2:
        logger.debug('ConvTranspose without bias')
        has_bias = False
        W = ensure_numpy_type(layers[node.input[1]])
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]])

    if len(W.shape) == 5:  # 3D conv
        raise NotImplementedError('Not implemented')

    elif len(W.shape) == 4:  # 2D conv
        W = W.transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        if params['group'] > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with groups != 1')

        if params['dilations'][0] > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')

        padding_mode = 'valid'
        pads = params['pads']
        strides = params['strides']

        if 2 * pads[0] + strides[0] - height == 0 and 2 * pads[1] + strides[1] - width == 0:
            padding_mode = 'same'

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(strides[0], strides[1]),
            padding=padding_mode,
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=node_name
        )

        layers[node_name] = input_0 = conv(input_0)

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        input_0.set_shape(input_0._keras_shape)

        if 'output_padding' in params and (params['output_padding'][0] > 0 or params['output_padding'][1] > 0):
            raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')

        if padding_mode == 'valid' and (pads[0] > 0 or pads[1] > 0):
            logger.debug('Add cropping layer for output padding')
            assert(len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=node_name + '_crop'
            )
            layers[node_name] = crop(input_0)
    else:
        raise AttributeError('Layer is not supported for now')


def convert_upsample(node, params, layers, node_name):
    """
        Convert upsampling layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available keras layers
        :param node_name: resulting layer name
        :return: None
    """
    logger = logging.getLogger('onnx2keras:upsample')

    if len(node.input) != 2:
        raise NotImplementedError('Upsample layer: Invalid Params')

    input_0 = ensure_tf_type(layers[node.input[0]])
    scale = ensure_numpy_type(layers[node.input[1]])[2:]

    mode = str(params['mode'])[2:-1]
    if mode == 'linear':
        mode = 'bilinear'

    upsample = keras.layers.UpSampling2D(size=(scale[0], scale[1]), interpolation=mode, name=node_name)

    layers[node_name] = upsample(input_0)

