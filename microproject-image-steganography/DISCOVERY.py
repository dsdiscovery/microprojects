import pandas as pd
from PIL import Image

def df_image(fileName):
  data = loadImage(fileName)
  return createImageDataFrame(data)

def loadImage(fileName, resize=False, format="RGB"):
  # Open the image using the PIL library
  image = Image.open(fileName)

  # Convert it to an (x, y) array:
  return imageToArray(image, format, resize)


# Resize the image to an `outputSize` x `outputSize` square, where `outputSize` is defined (globally) above.
def squareAndResizeImage(image, resize):
  import PIL

  w, h = image.size
  d = min(w, h)
  image = image.crop( (0, 0, d, d) ).resize( (resize, resize), resample=PIL.Image.LANCZOS )
  
  return image


# https://stackoverflow.com/questions/13405956/convert-an-image-rgb-lab-with-python
def rgb2lab(inputColor):
  num = 0
  RGB = [0, 0, 0]

  for value in inputColor:
    value = float(value) / 255

    if value > 0.04045:
      value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
    else :
      value = value / 12.92

    RGB[num] = value * 100
    num = num + 1

  XYZ = [0, 0, 0]

  X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
  Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
  Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
  XYZ[ 0 ] = round( X, 4 )
  XYZ[ 1 ] = round( Y, 4 )
  XYZ[ 2 ] = round( Z, 4 )

  XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
  XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
  XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

  num = 0
  for value in XYZ:
    if value > 0.008856:
      value = value ** ( 0.3333333333333333 )
    else:
      value = ( 7.787 * value ) + ( 16 / 116 )

    XYZ[num] = value
    num = num + 1

  Lab = [0, 0, 0]

  L = ( 116 * XYZ[ 1 ] ) - 16
  a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
  b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

  Lab [ 0 ] = round( L, 4 )
  Lab [ 1 ] = round( a, 4 )
  Lab [ 2 ] = round( b, 4 )

  return Lab


# Convert (and resize) an Image to an Lab array
def imageToArray(image, format, resize):
  import numpy as np

  w, h = image.size
  if resize:
    image = squareAndResizeImage(image, resize)

  image = image.convert('RGB')
  rgb = np.array(image)
  if format == "RGB":
    rgb = rgb.astype(int)
    return rgb.transpose([1,0,2])
  elif format == "Lab":
    lab = rgb.astype(float)
    for i in range(len(rgb)):
      for j in range(len(rgb[i])):
        lab[i][j] = rgb2lab(lab[i][j])
    return lab.transpose([1,0,2])
  else:
    raise Exception(f"Unknown format {format}")


imageCache = {}


def getTileImage(fileName, size):
  key = f"{fileName}-{size}px"

  if key not in imageCache:
    imageCache[key] = squareAndResizeImage(Image.open(fileName), size)

  return imageCache[key]



def isImageFile(file):
  for ext in [".jpg", ".jpeg", ".png"]:
    if file.endswith(ext) or file.endswith(ext.upper()):
      return True

  return False

def listTileImagesInPath(path):
  from os import listdir
  from os.path import isfile, join

  files = []
  for f in listdir(path + "/"):
    file = join(path + "/", f)
    if isfile(file) and isImageFile(file):
      files.append(file)

  return files


tada = "\N{PARTY POPPER}"


def run_test_case_1b(green_pixel):
  if len(green_pixel) != 1:
    print("\N{CROSS MARK} `green_pixel` must contain just one pixel.")
    return
  else:
    print("\u2705 `green_pixel` contains just one pixel!")

  if green_pixel["r"].sum() == 0 and green_pixel["g"].sum() == 255 and green_pixel["b"].sum() == 0:
    print("\u2705 `green_pixel` is a green pixel!")
    print(f"{tada} All tests passed! {tada}")
  else:
    print("\N{CROSS MARK} `green_pixel` looks like a pixel, but it's not green! Check your (x, y) coordinates.")
    return


def run_test_case_2(red, green, blue):
  import numbers  
  if isinstance(red, numbers.Number):
    print("\u2705 `red` is a number!")
  else:
    print(f"\N{CROSS MARK} `red` must be a number -- but yours is a {type(red)}.")
    return

  if red == 255:
    print("\u2705 `red` has the correct value!")
  else:
    print(f"\N{CROSS MARK} `red` is not the correct value.  (Did you use the orange pixel?)")
    return


  if isinstance(green, numbers.Number):
    print("\u2705 `green` is a number!")
  else:
    print(f"\N{CROSS MARK} `green` must be a number -- but yours is a {type(green)}.")
    return

  if green == 85:
    print("\u2705 `green` has the correct value!")
  else:
    print(f"\N{CROSS MARK} `green` is not the correct value.  (Did you use the orange pixel?)")
    return


  if isinstance(blue, numbers.Number):
    print("\u2705 `blue` is a number!")
  else:
    print(f"\N{CROSS MARK} `blue` must be a number -- but yours is a {type(blue)}.")
    return

  if blue == 46:
    print("\u2705 `blue` has the correct value!")
  else:
    print(f"\N{CROSS MARK} `blue` is not the correct value.  (Did you use the orange pixel?)")
    return

  print(f"{tada} All tests passed! {tada}")


def run_test_case_3(f):
  df = f("sample.png")
  
  if not isinstance(df, pd.DataFrame):
    print(f"\N{CROSS MARK} Your function must return a DataFrame.")
    return


  for colName in ['r', 'g', 'b', 'x', 'y']:
    if colName not in df:
      print(f"\N{CROSS MARK} `df` must contain a variable (column) `{colName}`.")
      return

  print("\u2705 `df` looks good!")
  print(f"{tada} All tests passed! {tada}")


def run_test_case_4(findAverageColor):
  pixelData = [
    { "r": 0, "g": 0, "b": 0 },
    { "r": 0, "g": 0, "b": 0 },
    { "r": 3, "g": 6, "b": 9 },
  ]
  result = findAverageColor(pd.DataFrame(pixelData))
  
  for colName in ['avg_r', 'avg_g', 'avg_b']:
    if colName not in result:
      print(f"\N{CROSS MARK} Dictionary must contain the key `{colName}`.")
      return
    else:
      print(f"\u2705 Dictionary contain the key `{colName}`.")

  if result["avg_r"] == 1 and result["avg_g"] == 2 and result["avg_b"] == 3:
    print("\u2705 The values all appear correct!")
    print(f"{tada} All tests passed! {tada}")
  else:
    print(f"\N{CROSS MARK} Dictionary data is incorrect.")


def run_test_case_5(findImageSubset):
  rawPixelData = [
    # [0]           [1]           [2]           [3]
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],  # [0]
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0], [30, 60, 90] ],  # [1]
    [ [ 0,  0,  0], [ 0,  0,  0], [30, 60, 90], [30, 60, 90] ],  # [2]
    [ [ 0,  0,  0], [30, 60, 90], [30, 60, 90], [30, 60, 90] ],  # [3]
    [ [30, 60, 90], [30, 60, 90], [30, 60, 90], [30, 60, 90] ],  # [4]
    [ [30, 60, 90], [30, 60, 90], [30, 60, 90], [ 0,  0,  0] ],  # [5]
    [ [30, 60, 90], [30, 60, 90], [ 0,  0,  0], [ 0,  0,  0] ],  # [6]
    [ [30, 60, 90], [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],  # [7]
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],  # [8]
  ]

  d = []
  for x in range(len(rawPixelData)):
    for y in range(len(rawPixelData[0])):
      p = rawPixelData[x][y]
      d.append({"x": x, "y": y, "r": p[0], "g": p[1], "b": p[2]})
  pixelData = pd.DataFrame(d)

  def TEST_findImageSubset(f, x, y, w, h, expected):
    result = f(pixelData, x, y, w, h)
    if len(result) != w * h:
      print(f"\N{CROSS MARK} findImageSubset(image, x={x}, y={y}, width={w}, height={h}) must have {w * h} pixels.")
      print("== Your DataFrame ==")
      print(result)
      return False

    if len(result[ result.x < x ]) != 0:
      print(f"\N{CROSS MARK} findImageSubset(image, x={x}, y={y}, width={w}, height={h}) must have no pixels less than x={x}.")
      print("== Your DataFrame ==")
      print(result)
      return False

    if len(result[ result.x >= x + w ]) != 0:
      print(f"\N{CROSS MARK} findImageSubset(image, x={x}, y={y}, width={w}, height={h}) must have no pixels greater than or equal to x={x + w}.")
      print("== Your DataFrame ==")
      print(result)
      return False

    if len(result[ result.y < y ]) != 0:
      print(f"\N{CROSS MARK} findImageSubset(image, x={x}, y={y}, width={w}, height={h}) must have no pixels less than y={y}.")
      print("== Your DataFrame ==")
      print(result)
      return False

    if len(result[ result.y >= y + h ]) != 0:
      print(f"\N{CROSS MARK} findImageSubset(image, x={x}, y={y}, width={w}, height={h}) must have no pixels greater than or equal to y={y + h}.")
      print("== Your DataFrame ==")
      print(result)
      return False
    

    print(f"\u2705 Test case for findImageSubset(image, x={x}, y={y}, width={w}, height={h}) appears correct.")
    return True

  r = TEST_findImageSubset(findImageSubset, 0, 0, 2, 2, [0, 0, 0])
  if not r: return

  r = TEST_findImageSubset(findImageSubset, 2, 0, 2, 2, [7.5, 15, 22.5])
  if not r: return

  r = TEST_findImageSubset(findImageSubset, 2, 2, 2, 2, [30, 60, 90])
  if not r: return

  r = TEST_findImageSubset(findImageSubset, 5, 1, 2, 2, [90/4, 180/4, 270/4])
  if not r: return

  r = TEST_findImageSubset(findImageSubset, 5, 1, 3, 2, [90/8, 180/8, 270/8])
  if not r: return

  r = TEST_findImageSubset(findImageSubset, 5, 1, 4, 3, [90/12, 180/12, 270/12])
  if not r: return

  r = TEST_findImageSubset(findImageSubset, 1, 1, 1, 3, [90/12, 180/12, 270/12])
  if not r: return

  print(f"{tada} All tests passed! {tada}")


def run_test_case_6(findAverageImageSubsetColor):
  rawPixelData = [
    # [0]           [1]           [2]           [3]
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],  # [0]
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0], [30, 60, 90] ],  # [1]
    [ [ 0,  0,  0], [ 0,  0,  0], [30, 60, 90], [30, 60, 90] ],  # [2]
    [ [ 0,  0,  0], [30, 60, 90], [30, 60, 90], [30, 60, 90] ],  # [3]
    [ [30, 60, 90], [30, 60, 90], [30, 60, 90], [30, 60, 90] ],  # [4]
    [ [30, 60, 90], [30, 60, 90], [30, 60, 90], [ 0,  0,  0] ],  # [5]
    [ [30, 60, 90], [30, 60, 90], [ 0,  0,  0], [ 0,  0,  0] ],  # [6]
    [ [30, 60, 90], [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],  # [7]
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],  # [8]
  ]

  d = []
  for x in range(len(rawPixelData)):
    for y in range(len(rawPixelData[0])):
      p = rawPixelData[x][y]
      d.append({"x": x, "y": y, "r": p[0], "g": p[1], "b": p[2]})
  pixelData = pd.DataFrame(d)

  def TEST_findAverageImageSubsetColor(f, x, y, w, h, expected):
    result = f(pixelData, x, y, w, h)

    if result["avg_r"] != expected[0] or result["avg_g"] != expected[1] or result["avg_b"] != expected[2]:
      print(f"\N{CROSS MARK} Test case for findAverageImageSubsetColor(image, x={x}, y={y}, width={w}, height={h}) did not have the expected value.")
      
      r = result["avg_r"]
      g = result["avg_g"]
      b = result["avg_b"]
      print(f"  Your Result: avg_r={r}, avg_g={g}, avg_b={b}")

      r = expected[0]
      g = expected[1]
      b = expected[2]
      print(f"  Expected Result: avg_r={r}, avg_g={g}, avg_b={b}")
      return False
    else:
      print(f"\u2705 Test case for findAverageImageSubsetColor(image, x={x}, y={y}, width={w}, height={h}) appears correct.")
      return True
    
  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 0, 0, 2, 2, [0, 0, 0])
  if not r: return

  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 2, 0, 2, 2, [7.5, 15, 22.5])
  if not r: return

  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 2, 2, 2, 2, [30, 60, 90])
  if not r: return

  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 5, 1, 2, 2, [90/4, 180/4, 270/4])
  if not r: return

  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 5, 1, 3, 2, [15, 30, 45])
  if not r: return

  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 5, 1, 4, 3, [90/12, 180/12, 270/12])
  if not r: return

  r = TEST_findAverageImageSubsetColor(findAverageImageSubsetColor, 1, 1, 1, 3, [10, 20, 30])
  if not r: return

  print(f"{tada} All tests passed! {tada}")



def run_test_case_8(findBestTile):
  real_df = pd.DataFrame([
      {'file': 'notebook-images/test.png', 'r': 47.19722525581813, 'g': 49.03421116311881, 'b': 38.60877549417687},
      {'file': 'notebook-images/test2.png', 'r': 54.24409328969397, 'g': 59.3141053878179, 'b': 52.97987993308968},
      {'file': 'notebook-images/test3.png', 'r': 46.41423991872082, 'g': 47.89200069370779, 'b': 37.011986112075455}
  ])

  try:
    bestMatch = findBestTile(real_df, 0, 0, 0)
    assert(type(bestMatch) == type(pd.DataFrame())), "findBestMatch must return a DataFrame"
    assert(len(bestMatch) == 1), "findBestMatch must return exactly one best match"
    assert(bestMatch['file'].values[0] == 'notebook-images/test3.png'), "findBestMatch did not return the best match for test (r=0, g=0, b=0)"
    print(f"\u2705 Test case #1 (r=0, g=0, b=0) passed!")

    bestMatch = findBestTile(real_df, 47, 49, 38)
    assert(bestMatch['file'].values[0] == 'notebook-images/test.png'), "findBestMatch did not return the best match for test (r=47, g=49, b=38)"
    print(f"\u2705 Test case #1 (r=47, g=49, b=38) passed!")

    bestMatch = findBestTile(real_df, 54, 49, 38)
    assert(bestMatch['file'].values[0] == 'notebook-images/test.png'), "findBestMatch did not return the best match for test (r=54, g=49, b=38)"
    print(f"\u2705 Test case #1 (r=54, g=49, b=38) passed!")

    bestMatch = findBestTile(real_df, 54, 49, 52)
    assert(bestMatch['file'].values[0] == 'notebook-images/test2.png'), "findBestMatch did not return the best match for test (r=54, g=49, b=52)"
    print(f"\u2705 Test case #1 (r=54, g=49, b=52) passed!")

    bestMatch = findBestTile(real_df, -100, -100, -100)
    assert(bestMatch['file'].values[0] == 'notebook-images/test3.png'), "findBestMatch did not return the best match for test (r=-100, g=-100, b=-100)"
    print(f"\u2705 Test case #1 (r=-100, g=-100, b=-100) passed!")

    print(f"{tada} All tests passed! {tada}")

  except AssertionError as e:
    print(f"\N{CROSS MARK} {e}.")

  


# def createImageDataFrame(width, height):
#   import random

#   data = []
#   for x in range(width):
#     for y in range(height):
#       data.append( {"x": x, "y": y, "r": random.randint(0, 255), "g": random.randint(0, 255), "b": random.randint(0, 255)} )
  
#   return pd.DataFrame(data)


def createImageDataFrame(img):
  data = []
  width = len(img)
  height = len(img[0])

  for x in range(width):
    for y in range(height):
      pixel = img[x][y]
      r = pixel[0]
      g = pixel[1]
      b = pixel[2]

      d = {"x": x, "y": y, "r": r, "g": g, "b": b}
      data.append(d)  

  return pd.DataFrame(data)


def saveDataFrameAsImage(df, fileName):
  width = max(df.x) + 1
  height = max(df.y) + 1

  image = Image.new('RGB', (width, height))
  for index, row in df.iterrows():
    image.paste( (row.r, row.g, row.b), (row.x, row.y, row.x + 1, row.y + 1) )
  image.save(fileName)