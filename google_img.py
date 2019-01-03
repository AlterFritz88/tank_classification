from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"tank syrian war", "limit":100, 'format' : 'jpg'}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images


'''
https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
'''