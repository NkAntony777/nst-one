import streamlit as st
import argparse
import os
import sys
import PIL
import yaml
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image

from src.process_image import load_image, get_image_name_ext
from src.train_model import train_image


def image_style_transfer(config):
    """Implements neural style transfer on a content image using a style image, applying provided configuration."""
    if config.get('image_dir') is not None:
        image_dir = config.get('image_dir')
        content_path = os.path.join(image_dir, config.get('content_filename'))
        style_path = os.path.join(image_dir, config.get('style_filename'))
        output_dir = config.get('output_dir') if config.get(
            'output_dir') is not None else image_dir
    else:
        output_dir = config.get('output_dir')
        content_path = config.get('content_filepath')
        style_path = config.get('style_path')

    verbose = not config.get('quiet')

    if verbose:
        print("Loading content and style images...")

    try:
        content_img = Image.open(content_path)
    except FileNotFoundError:
        print(f"ERROR: could not find such file: '{content_path}'.")
        return
    except PIL.UnidentifiedImageError:
        print(f"ERROR: could not identify image file: '{content_path}'.")
        return

    try:
        style_img = Image.open(style_path)
    except FileNotFoundError:
        print(f"ERROR: could not find such file: '{style_path}'.")
        return
    except PIL.UnidentifiedImageError:
        print(f"ERROR: could not identify image file: '{style_path}'.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load content and style images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = config.get('output_image_size')
    if output_size is not None:
        if len(output_size) > 1:
            output_size = tuple(output_size)
        else:
            output_size = output_size[0]

    content_tensor = load_image(content_path, device, output_size=output_size)
    output_size = (content_tensor.shape[2], content_tensor.shape[3])
    style_tensor = load_image(style_path, device, output_size=output_size)

    if verbose:
        print("Content and style images successfully loaded.")
        print()
        print("Initializing output image...")

    # initialize output image
    generated_tensor = content_tensor.clone().requires_grad_(True)

    if verbose:
        print("Output image successfully initialized.")
        print()

    # load training configuration if provided
    
    train_config = dict()
    if (train_config_path := config.get('train_config_path')) is not None:
        if verbose:
            print("Loading training configuration file...")

        try:
            with open(train_config_path, 'r') as f:
                train_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"ERROR: could not find such file: '{train_config_path}'.")
            return
        except yaml.YAMLError:
            print(f"ERROR: fail to load yaml file: '{train_config_path}'.")
            return

        if verbose:
            print("Training configuration file successfully loaded.")
            print()

    if verbose:
        print("Training...")

    content_img_name, content_img_fmt = get_image_name_ext(content_path)
    style_img_name, _ = get_image_name_ext(style_path)

    output_img_fmt = config.get('output_image_format')
    if output_img_fmt == 'same':
        output_img_fmt = content_img_fmt

    # train model
    success = train_image(content_tensor, style_tensor, generated_tensor, device, train_config,
                          output_dir, output_img_fmt, content_img_name, style_img_name, verbose=verbose)

    # save output image to specified directory
    if success:
        save_image(generated_tensor, os.path.join(
            output_dir, f'nst-{content_img_name}-{style_img_name}-final.{output_img_fmt}'))

    if verbose:
        print(
            f"Output image successfully generated as {os.path.join(output_dir, f'nst-{content_img_name}-{style_img_name}-final.{output_img_fmt}')}.")


def main():
    st.title("Neural Style Transfer")

    # Sidebar for user input
    st.sidebar.title("Input Configuration")
    image_dir = st.sidebar.text_input("Image Directory", "")
    content_filename = st.sidebar.text_input(
        "Content Image Filename", "content.jpg")
    style_filename = st.sidebar.text_input("Style Image Filename", "style.jpg")
    content_filepath = st.sidebar.text_input("Content Image Path", "")
    style_filepath = st.sidebar.text_input("Style Image Path", "")
    output_dir = st.sidebar.text_input("Output Directory", "")
    output_image_size = st.sidebar.text_input(
        "Output Image Size (width height)", "")
    output_image_format = st.sidebar.selectbox(
        "Output Image Format", ["jpg", "png", "jpeg", "same"])
    train_config_path = st.sidebar.text_input("Training Config Path", "")
    quiet = st.sidebar.checkbox("Quiet Mode")

    args = argparse.Namespace(
        image_dir=image_dir,
        content_filename=content_filename,
        style_filename=style_filename,
        content_filepath=content_filepath,
        style_filepath=style_filepath,
        output_dir=output_dir,
        output_image_size=output_image_size,
        output_image_format=output_image_format,
        train_config_path=train_config_path,
        quiet=quiet
    )

    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            image_style_transfer(config)
        st.success("Image Generated Successfully!")


if __name__ == '__main__':
    main()
