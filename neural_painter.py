#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: neural_painter.py
# $Date: Thu Apr 07 10:39:34 2016 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import argparse


import theano
import theano.tensor as T

import os
import numpy as np
import cv2


eps = 1e-8

NONLIN_TABLE = dict(
    relu=T.nnet.relu,
    tanh=T.tanh,
    abs_tanh=lambda x: abs(T.tanh(x)),
    sigmoid=T.nnet.sigmoid,
    softplus=T.nnet.softplus,
    sin=T.sin,
    cos=T.cos,
    sgn=T.sgn,
    sort=lambda x: T.sort(x, axis=1),
    abs=abs,
    log_abs=lambda x: T.log(abs(x) + eps),  # this is awesome
    log_abs_p1=lambda x: T.log(abs(x) + 1),
    log_relu=lambda x: T.log(T.nnet.relu(x) + eps),
    log_square=lambda x: T.log(x**2 + eps),  # just a scalar

    xlogx_abs=lambda x: T.xlogx.xlogx(abs(x) + eps),
    xlogx_abs_p1=lambda x: T.xlogx.xlogx(abs(x) + 1),
    xlogx_relu=lambda x: T.xlogx.xlogx(T.nnet.relu(x) + eps),
    xlogx_relu_p1=lambda x: T.xlogx.xlogx(T.nnet.relu(x) + 1),
    xlogx_square=lambda x: T.xlogx.xlogx(x**2 + eps),

    softmax=T.nnet.softmax,
    logsoftmax=T.nnet.logsoftmax,
    hard_sigmoid=T.nnet.hard_sigmoid,
    identity=lambda x: x,
    square=lambda x: x**2
)


def get_func(rng, nonlin, hidden_size=100, nr_hidden=3,
             input_dim=2,
             output_dim=1, recurrent=False,
             output_nonlin=lambda x: x,
             use_bias=True,
             std=1, mean=0):
    '''return function of [0,1]^2 -> intensity \in [0, 1]^c '''
    coords = T.matrix()
    v = coords

    def get_weights(shape):
        W = theano.shared(rng.randn(*shape) * std + mean)
        if use_bias:
            b = theano.shared(rng.randn(shape[1]) * std + mean)
        else:
            b = theano.shared(np.zeros(shape[1]))
        return W, b

    def apply_linear(v, W, b):
        '''Wx + b'''
        return T.dot(v, W) + b.dimshuffle('x', 0)

    def make_linear(v, shape):
        W, b = get_weights(shape)
        return apply_linear(v, W, b)

    v = make_linear(v, (input_dim, hidden_size))
    v = nonlin(v)

    hidden_shape = (hidden_size, hidden_size)
    W, b = None, None
    for i in range(nr_hidden):
        if W is None or not recurrent:
            W, b = get_weights(hidden_shape)
        v = apply_linear(v, W, b)
        v = nonlin(v)

    v = make_linear(v, (hidden_size, output_dim))
    v = output_nonlin(v)
    v = (v - v.min(axis=0, keepdims=True)) / (
        v.max(axis=0) - v.min(axis=0) + 1e-8).dimshuffle('x', 0)

    return theano.function([coords], v)


def draw(func, w, h, coord_bias=False):
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))[::-1],
                      dtype='float32').reshape((2, -1)).swapaxes(0, 1) / [w, h]

    if coord_bias:
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)
    coords = coords.astype('float32')

    img = (func(coords).reshape((w, h, -1)) * 255).astype('uint8')
    if img.shape[2] == 1:
        img = img[:,:]
    return img


def cvpause():
    while True:
        if (cv2.waitKey(0) & 0xff) == ord('q'):
            break
        print('press `q` to close this window')


def get_nonlin(name, rng):
    if name == 'random_every_time':
        def nonlin(x):
            return NONLIN_TABLE[rng.choice(list(NONLIN_TABLE))](x)
        return nonlin

    if name == 'random_once':
        return NONLIN_TABLE[rng.choice(list(NONLIN_TABLE))]

    return NONLIN_TABLE[name]


def sanitize_str(x):
    x = x.replace('/', '-')
    i = 0
    while i < len(x) and x[i] == '-':
        i += 1
    return x[i:]


def args2name(args):
    black_list = ['output', 'auto_name']

    return '-'.join(['{}:{}'.format(key, sanitize_str(str(value)))
                     for key, value in sorted(args._get_kwargs())
                     if key not in black_list and value is not None])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--image_size', help='wxh', default='100x100')
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--nr_hidden', default=3, type=int)
    parser.add_argument('--recurrent', action='store_true')
    parser.add_argument('--coord_bias', action='store_true')
    parser.add_argument('--nr_channel', default=1, type=int, choices={1, 3})
    parser.add_argument('--nonlin', default='tanh',
                        choices=list(NONLIN_TABLE) + [
                            'random_once', 'random_every_time'])

    parser.add_argument('--output_nonlin', default='identity',
                        choices=list(NONLIN_TABLE))
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--use_bias', action='store_true',
                        help='use bias in hidden layer')
    parser.add_argument('--batch_norm_position',
                        choices={'before_nonlin', 'after_nonlin'},
                        default='before_nonlin')
    parser.add_argument('--output', '-o', help='output image path')
    parser.add_argument('--auto_name', action='store_true',
                        help='append generation parameters'
                        ' to the name of the output')

    return parser.parse_args()


def run(args):
    rng = np.random.RandomState(args.seed)

    w, h = map(int, args.image_size.split('x'))

    nonlin = get_nonlin(args.nonlin, rng)
    output_nonlin = get_nonlin(args.output_nonlin, rng)


    if args.batch_norm:
        batch_norm=lambda x: (x - T.mean(x, axis=1, keepdims=True)) / T.std(
            x, axis=1, keepdims=True)
        def add_bn(nonlin):
            def func(x):
                if args.batch_norm_position == 'before_nonlin':
                    x = batch_norm(x)
                x = nonlin(x)
                if args.batch_norm_position == 'after_nonlin':
                    x = batch_norm(x)
                return x
            return func
        nonlin = add_bn(nonlin)

    input_dim = 2
    if args.coord_bias:
        input_dim += 1

    print('Compiling...')
    func = get_func(rng, nonlin, hidden_size=args.hidden_size,
                    nr_hidden=args.nr_hidden,
                    input_dim=input_dim,
                    output_dim=args.nr_channel,
                    recurrent=args.recurrent,
                    output_nonlin=output_nonlin,
                    use_bias=args.use_bias)

    print('Drawing...')
    img = draw(func, w, h, coord_bias=args.coord_bias)

    if args.output:
        output = args.output
        name, ext = os.path.splitext(output)
        if args.auto_name:
            name = name + '-' + args2name(args)
        cv2.imwrite(name + ext, img)
    else:
        cv2.imshow('img', img)
        cvpause()


def main():
    run(get_args())


if __name__ == '__main__':
    main()


# vim: foldmethod=marker
