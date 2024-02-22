# Attacking-an-Artificial-Neural-Network-ANN-

# README: Attacking an Artificial Neural Network (ANN)

## Introduction

Inspired by the human brain, Artificial Neural Networks (ANNs) are a type of computer vision model designed to classify images into specific categories. For the purpose of this project, we're focusing on ANNs that recognize handwritten digits (ranging from 0 to 9) from black-and-white images with a resolution of 28x28 pixels.

The core objective of this project is to not only compute an ANN output for a given input but also to develop functions that "attack" the ANN. These attacks aim to generate inputs that can deceive the neural network into making incorrect classifications.

## Main Functions

### `adversarial_image`

This function aims to find an adversarial image for the given image, weights, and biases files.

**Parameters**:

- `image_file_name`: A string representing the filename of the image.
- `weights_file_name`: A string representing the filename of the weights.
- `biases_file_name`: A string representing the filename of the biases.

**Returns**:

A list of integers representing the adversarial image. If the algorithm fails to find an adversarial image, it returns the list `[-1]`.

**Usage**:
```python
x1 = read_image('image.txt')
x2 = adversarial_image('image.txt', 'weights.txt', 'biases.txt')
if x2[0] == -1:
    print('Algorithm failed.')
else:
    write_image(x2, 'new_image.txt')
    q = compute_difference(x1, x2)
    print(f'An adversarial image is found! Total of {q} pixels were flipped.')
```

### `write_image`

This function writes the image represented by the list to a specified file.

**Parameters**:

- `x`: A list of integers representing the image.
- `file_name`: A string representing the desired filename.

**Returns**:

Writes out each pixel represented in the list `x` to a file with the name `file_name` as a 28x28 image.

**Usage**:
```python
x = read_image('image.txt')
x = modified_list(238, x)
x = modified_list(210, x)
write_image(x, 'new_image.txt')
```

**Algorithm Overview**:

1. Open the specified file in write mode.
2. Iterate through each line (for 28 lines).
3. For each line, iterate through each pixel (for 28 pixels).
4. Write the pixel to the file.
5. After writing all 28 pixels for a line, insert a newline character to move to the next line.
6. Close the file after writing all the lines and pixels.

## Getting Started

1. Ensure you have the necessary dependencies installed.
2. Download the project files.
3. Use the functions as demonstrated above to generate adversarial images or write images to files.

## Conclusion

By utilizing the above functions, users can effectively attempt to attack a given ANN and identify potential vulnerabilities in its image recognition capabilities. This exploration is crucial in understanding and enhancing the robustness of neural networks against adversarial attacks.
