"""
Author: <Dhivyan Deepak Sureshkumar> (<32466951>)
"""

def linear(x, w, b): 
    
    return sum(w[j]*x[j] for j in range(len(w))) + b


def linear_layer(x, w, b): 

    return [linear(x, w[i], b[i]) for i in range(len(w))]


def inner_layer(x, w, b):
 
    return [max(linear(x, w[i], b[i]), 0.0) for i in range(len(w))]


def inference(x, w, b): 
 
    num_layers = len(w)
    
    for l in range(num_layers-1):
        x = inner_layer(x, w[l], b[l])
        
    return linear_layer(x, w[num_layers-1], b[num_layers-1])


def read_weights(file_name):

    weights_file = open(file_name,"r")
    w = []
    for line in weights_file:
        if "#" == line[0]:
            w.append([])
        else:
            w[-1].append([float(w_ij) for w_ij in line.strip().split(",")])
    
    return w


def read_biases(file_name): 

    biases_file = open(file_name,"r")
    b = []
    for line in biases_file:
        if not "#" == line[0]:
            b.append([float(b_j) for b_j in line.strip().split(",")])
    
    return b


def read_image(file_name): 

    image_file = open(file_name,"r")
    x = []
    for line in image_file:
        for x_i in line.strip():
            x.append(int(x_i))
            
    return x


def argmax(x): 

    num_inputs = len(x)
    max_index = 0
    
    for i in range(1,num_inputs):
        if x[max_index] < x[i]:
            max_index = i
            
    return  max_index


def predict_number(image_file_name, weights_file_name, biases_file_name): 

    x = read_image(image_file_name)
    w = read_weights(weights_file_name)
    b = read_biases(biases_file_name)
    
    y = inference(x, w, b)
    return argmax(y)

''' end of functions used with Assignment1 solutions '''

def flip_pixel(x):
    '''
    Input: An integer (x) representing a pixel in the image.
    Output: An integer representing the flipped pixel.
    The function flip pixel(x) must behave as follows:
    >>> x = 1
    >>> flip_pixel(x)
    0
    >>> x = 0
    >>> flip_pixel(x)
    1

    This is an algorithm which is used to flip a single pixel.

    This is calculated by taking an input which must be a binary number
    (either 1 or 0), and then outputing 1 if 0 was inputted or outputing 0 if 1
    was inputed.

    In my implementation i used two if statements to check whether the inputted
    value was either a 1 or 0. Then within the if statements if the inputted
    value equalled 1 i returned 0 and if the inputted value equalled 0 i
    returned 1. Thus acheiving the goal of the algorithm flip_pixel
    
    '''
    if x == 1:
        return 0
    if x == 0:
        return 1


def modified_list(i,x):
    '''
    Input: A list of integers (x) representing the image and an integer (i) representing the position (i.e., index) of the pixel.
    Output: A list of integers (x) representing the modified image.
    For example, the function modified list(i,x) behaves as follows for input list x and index i:
    >>> x = [1, 0, 1, 1, 0, 0, 0]
    >>> i = 2
    >>> modified_list(i,x)
    [1, 0, 0, 1, 0, 0, 0]
    >>> x = [0, 0, 0, 0, 0]
    >>> i = 0
    >>> modified_list(i,x)
    [1, 0, 0, 0, 0]
    

    This is an algorithm that is used used to flip the pixel that is located
    at position i in list x

    This is calculated by creating a copy of x, so that the original list of
    integers is not modified. Then flipping the pixel at index i of the copied
    list. We then output this list to create the new modified list with the
    pixel at index i flipped.

    In my implementation i used slicing to create a copy of x since the use of
    the copy function was not allowed as listed in the assignment specifications.
    By then using indexing i changed the pixel at index i to its flipped pixel,
    by calling the function flipped_pixel. By assigning the pixel at index i
    to equal to flipped pixel at index i, i was able to acheive the goal of the
    algorithm to flip the pixel that is located at position i in list x
    '''
    x1= x[:]
    x1[i] = flip_pixel(x1[i])
    return x1

def compute_difference(x1,x2):
    '''
    Input: A list of integers (x1) representing the input image and a list of integers (x2) representing the adversarial image.
    Output: An integer representing the total absolute difference between the elements of x1 and x2.
    For example, the function compute difference(x1,x2) behaves as follows for lists x1 and x2:
    >>> x1 = [1, 0, 1, 1, 0, 0, 0]
    >>> x2 = [1, 1, 1, 0, 0, 0, 1]
    >>> compute_difference(x1,x2)
    3
    >>> x1 = [1, 0, 1, 1]
    >>> x2 = [1, 0, 1, 1]
    >>> compute_difference(x1,x2)
    0

    This is an algorithm that is used to compute the total absolute difference
    between the adversarial image and the original image

    This is calculated by iterating through all integers in the lists x1 & x2
    and checking if the element at index i in x1 is equal to the corresponding
    element at index i in x2. If they are not equal, the counter is increased
    by 1 so that the varaible counter in the end will equal total absolute
    difference between the adversarial image and the original image.

    In my implementation i first created a counter variable that is equal to
    0. I then used a for loop to interate through all the intergers in list x1
    and x2. Since the lengths of the lists are the same, iterating through the
    length of list x1 will also iterate through the same indexes in x2. By then
    using a if statment i checked if the element at index i in x1 was the same
    as index i in x2. If this if statement passed, the counter will increase by
    1. At the end of the function i returned the count which will output the
    total absolute difference between the adversarial image and the original
    image.
    '''
    count = 0
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            count += 1
    return count


def select_pixel(x, w, b):
    '''
    Input: A list of inputs (x), a list of tables of weights (w) and a table of biases (b).
    Output: An integer (i) either representing the pixel that is selected to be flipped, or with value -1 representing
    no further modifications can be made.
    For example, the function select pixel(x, w, b) can behave as follows for input list x, list of tables of weights
    w and table of biases b:
    >>> x = read_image('image.txt')
    >>> w = read_weights('weights.txt')
    >>> b = read_biases('biases.txt')
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    238
    >>> x = modified_list(pixel,x)
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    210
    >>> x = modified_list(pixel,x)
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    238

    This is an algorithm that is used to either select which pixel to flip
    or conclude that no further modifications can be made. The pixel that is
    to be flipped should be selected based on its overall impact on the output list
    of the ANN. In my implementation I calcualated the overall impact by decreasing
    the most likely number as much as possible and add this difference to the increase
    of the second most likely number. For example in this test case, modifying pixel at
    position 238 decreases the score for number 4 by around 0.8 while increasing the score
    for number 9 (i.e., the second most likely number) bya round 0.3. Therefore the overall
    impact of flipping pixel at position 238 would be around 1.1.

    This is calculated by using inference by inputting (x), a list of tables of weights (w)
    and a table of biases (b), to create a list of predictions. Then the function must
    calcualate the most likely and second most likely predicted scores from the list of
    predictions. Then a list of predictions must be calculated for each possible modified
    image,where it iterates through each pixel and modifies it to create a new modified
    image. Then the overall impact must be calculated by using the original predicted scores
    lists maximum value and subtrating that with the modified images predicted score in
    the same index as the maximum value from the original list. This is then added to the number
    at the index of second most likely number from the original lists predicted scores subtracted with the
    original predicted scores list 2nd most likely number. This will calculate the overall impact for
    each modified pixel in the image. We then need to find the maximum impact as this will
    give us the pixel that needs to be changed in order to get the biggest impact. The function
    then checks if the maximum difference is less than 0 as if it is the output
    needs to be -1 since, the following specification states that the value -1
    represents no further modifications can be made. (i.e., when the overall impact of flipping
    is negative for all pixels). Finally the function outputs the index of which pixel
    when changed causes the greatest overall impact, or outputs -1 if no further modifications
    can be made.

    In my implementation, I first called the function inference to generate the predicted
    scores from image x. I then caluclated the most likely outcome using the max function,
    and also recorded its index in the variables maxval and maxindex respectivly. I them made
    a copy of the list and remove its maximum value so I could determine its second most likely
    score using the max function. I then used a for loop to interate through all pixels in image
    x. By creating a new varabile a_list which holds image x with a flipped pixel at index i, using
    the function modified list i was able to create images with each pixel in the original image
    flipped. I then called to the inference function within the loop to generate the predicted
    scores of each modifed image. By then assigning the scores at the indexes of the orginal predicted
    scores [maxindex] and [maxindex2], I am able to compare the impacts using (maxval - a_maxval) +
    (a_maxval2 - maxval2) for each flipped pixel in the image. I then used the a if statement to
    see if the difference after each iteration is greater than the highest difference from previous
    modified lists. If the impact was greater then it would be assignmed to max_diff. So that
    after the loop is complete max_diff will hold the greatest impact and then the pixel_index will
    hold the index which created the most impact. I then used a if statement to check if the max_diff
    is negative. If it is negative the function should output -1 since the overall impact is negative
    for all possible modified lists. 
    
     '''
    max_diff = -1
    pixel_index = 0

    prediction_list = inference(x, w, b)
    maxval = max(prediction_list)
    maxindex = argmax(prediction_list)
    
    listcopy = prediction_list.copy()
    listcopy.remove(maxval)

    maxval2 = max(listcopy)
    maxindex2 = prediction_list.index(maxval2)
    
    for i in range(len(x)):
        a_list = modified_list(i,x)
        a_prediction = inference(a_list, w, b)
        a_maxval = a_prediction[maxindex] 
        a_maxval2 = a_prediction[maxindex2] 

        difference = (maxval - a_maxval) + (a_maxval2 - maxval2)
        if difference > max_diff:
            max_diff = difference
            pixel_index = i
    if max_diff < 0: return -1
    else: return pixel_index
    

def write_image(x, file_name):
    '''
    Input: A list of integers (x) representing the image and a string (file name) representing the file name.
    Output: Write out each pixel represented in the list x to a file with the name file name as a 28x28 image.
    For example, the function write image(x, file name) behaves as follows for input list x and file name file name:
    >>> x = read_image('image.txt')
    >>> x = modified_list(238,x)
    >>> x = modified_list(210,x)
    >>> write_image(x,'new_image.txt')

    This algorithm is to write the list x into a file with name file name as
    a 28x28 image

    This is completed by iterating through each line and then each pixel
    within the file_name. Iterating through each pixel can be done by
    using indexing and a counter. To create new lines in a 28x28 image,
    the /n strig is used every 28 pixels to generate a new line so the image
    can be formed. When the function reaches te last pixel it will skip writing
    a new line since this will bring the cursor onto the next line, then the
    program closes the file since the image has now been generated.

    In my implementation, i first defined wfile as the file that the image
    will be written into. I then used a for loop to iterate through each
    line in the image x and then a nested for loop to iterate through each
    pixel in each line. I used range(0,28) in my for loops since the image
    is a 28x28 file. I then used the function write that writes the string
    form of each pixel, this iterates through all the pixels as the count
    varaible is used as the index and this is increased by one after every loop.
    I then used an if statement to check if the count was not equal to 784. If not
    it will write a new line using [\n], after every 28 pixels as it is contained
    within the outer for loop. The reason i used the for loop was to ensure the
    new line wasnt written at the end of the image. Thus stopping it from writing
    an additonal new line and instead close the file since all pxiels of the image
    have been written.
    
    '''
    wfile = open(file_name,'w')
    count = 0
    for i in range(0,28):
        for j in range(0,28):
            wfile.write(str(x[count]))
            count += 1
        if count != 784:
            wfile.write('\n')
    wfile.close

def adversarial_image(image_file_name,weights_file_name,biases_file_name):
    '''
    Input: A string (i.e., image file name) that corresponds to the image file name, a string (i.e., weights file name) that corresponds to the weights file name and a string (i.e., biases file name) that corresponds to the biases file name.
    Output: A list of integers representing the adversarial image or the list [-1] if the algorithm is unsuccesful in finding an adversarial image.
    For example, adversarial image can behave as follows for file names ‘image.txt’, ‘weights.txt’ and ‘biases.txt’:
    >>> x1 = read_image('image.txt')
    >>> x2 = adversarial_image('image.txt','weights.txt','biases.txt')
    >>> if x2[0] == -1:
    ...    print('Algorithm failed.')
    ... else:
    ...    write_image(x2,'new_image')
    ...    q = compute_difference(x1,x2)
    ...    print('An adversarial image is found! Total of ' + str(q) + ' pixels were flipped.')
    ...
    An adversarial image is found! Total of 2 pixels were flipped.

    This algorithm is used to solve the adversarial image generation task by generating
    an adversarial image based on an input image 

    This is completed by first checking if the select pixels output is -1 meaning that
    no further modifications can be made. If no further modifications can be made
    the adversarial image function should output the list [-1]. I then make a copy of
    the image x as i do not want to modify this image i then implemented a modified list
    which modifies the pixel that the select pixel ouputs (the pixel with the greatest
    impact). However this only modifies the image once, this is why I repeated this
    untill the select pixel starts outputting the same pixel. By adding all the modified images
    to the list i was easily able to access the last modifeid image so it too can be modified.
    To solve the adversarial image generaion task I ouputted the value at the second last index

    In my implementation i defined x, w, b as the functions readimage, readweights and readbiases
    of the file_name respectivly. I then called the select pixel function using the image,weieghts
    and biases file as inputs to generate the pixel with the biggest impact in the image x. I then
    checked if it was equal to -1, if so i returned the list [-1] as required by the assignment
    specifications. I then made a copy of x using splicing so that the original image file does
    not get altered. I then used the modified list function which used the select pixel function as an
    input and the copy of the image x. This will modify the image by flipping the pixel with the most impact.
    I then used a while loop which runs until the pixel with the greatest impact from the last and second last
    images fron the image_list are the same. This is done using the selectpixel function as well as the indicies
    [-1] and [-2]. Within the while loop, i use the select_pixel function to find the most pixel with the highest
    impact for the last image in the image list. In this case the most recent modified image. I then appended the
    new modified image which flips the pixel with the highest impact of the last image in image_list. Once the while
    loop is done (the function starts flipping the same pixel over and over). I output the second last image in the
    image file as this would be the i In thmage which hasnt had the same pixel flipped twice. Therfore is the
    image where no further modifications can be made. Thus completing the adversrial image generation task.
    
    ''' 
    
    x , w , b = read_image(image_file_name) , read_weights(weights_file_name) , read_biases(biases_file_name)
    pixel = select_pixel(x, w, b)
    if pixel == -1:
        return [-1]
    x1 = x[:]
    modlist = modified_list(pixel, x1)
    image_list = [x, modlist]


    while select_pixel(image_list[-1], w, b) != select_pixel(image_list[-2], w , b):
        pixel = select_pixel(image_list[-1], w, b)
        image_list.append(modified_list(pixel, image_list[-1]))
    adverse_image = image_list[-2]
    return adverse_image


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
