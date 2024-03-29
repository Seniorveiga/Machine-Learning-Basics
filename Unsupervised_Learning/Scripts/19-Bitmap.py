import pandas as pd
# Import pyplot
from matplotlib import pyplot as plt

lcd_digits_csv = pd.read_csv("lcd_digits.csv")
lcd_digits = pd.DataFrame(lcd_digits_csv)

# Select the 0th row: digit
digit = lcd_digits[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape((13,8))

"""
En castellano, lo que tienes es un csv que tiene las imagenes como si fuesen una sola fila con muchas columnas, total 
tienes 100 filas con 104 columnas. Lo que hacemos es como si cogiésemos una de esas filasy la ponemos de forma 
matricial y eso revela un mapa de bits.

Por eso la forma es (13,8), porque cada imagen son 104 columnas, que se reparte a su vez en 13 filas 8 columnas.
"""
# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

#---------Learning the part of the image-------------------
# Import NMF
from sklearn.decomposition import NMF

def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Create an NMF model: model
model = NMF(n_components = 7)

# Apply fit_transform to samples: features
features = model.fit_transform(lcd_digits)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)

"""
You cannot use this with PCA as it is not possible to sepparate the data.
"""