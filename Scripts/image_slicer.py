
import os
from PIL import Image
from math import sqrt, ceil, floor



class Tile(object):
    """Represents a single tile."""
    def __init__(self, image, number, position, coords, filename=None):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        self.filename = filename

    @property
    def row(self):
        return self.position[0]

    @property
    def column(self):
        return self.position[1]

    @property
    def basename(self):
        """Strip path and extension. Return base filename."""
        return get_basename(self.filename)

    def generate_filename(
        self, directory=os.getcwd(), prefix="tile", format="png", path=True
    ):
        """Construct and return a filename for this tile."""
        filename = prefix + "_{col:02d}_{row:02d}.{ext}".format(
            col=self.column, row=self.row, ext=format.lower().replace("jpeg", "jpg")
        )
        if not path:
            return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format="png"):
        if not filename:
            filename = self.generate_filename(format=format)
        self.image.save(filename, format)
        self.filename = filename

    def __repr__(self):
        """Show tile number, and if saved to disk, filename."""
        if self.filename:
            return "<Tile #{} - {}>".format(
                self.number, os.path.basename(self.filename)
            )
        return "<Tile #{}>".format(self.number)


def calc_columns_rows(n):
    """
    Calculate the number of columns and rows required to divide an image
    into ``n`` parts.

    Return a tuple of integers in the format (num_columns, num_rows)
    """
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)


def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)


def join(tiles,row=0,col=0, width=0, height=0):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @param ``width`` - Optional, width of combined image.
    @param ``height`` - Optional, height of combined image.
    @return ``Image`` instance.
    """
    # Don't calculate size if width and height are provided
    # this allows an application that knows what the
    # combined size should be to construct an image when
    # pieces are missing.

    if width > 0 and height > 0:
        im = Image.new("RGBA", (width, height), None)
    else:
        #im = Image.new("RGBA", get_combined_size(tiles), None)
        total_size = tiles[0].image.size[0] * col, tiles[0].image.size[1] * row #(width*cols , height*rows)
        im = Image.new("RGBA", total_size, None)

    #columns, rows = calc_columns_rows(len(tiles))
    columns=col
    rows=row
    for tile in tiles:
        try:
            im.paste(tile.image, tile.coords)
        except IOError:
            # do nothing, blank out the image
            continue
    return im


def validate_image(image, number_tiles):
    """Basic sanity checks prior to performing a split."""
    TILE_LIMIT = 99 * 99

    try:
        number_tiles = int(number_tiles)
    except BaseException:
        raise ValueError("number_tiles could not be cast to integer.")

    if number_tiles > TILE_LIMIT or number_tiles < 2:
        raise ValueError(
            "Number of tiles must be between 2 and {} (you \
                          asked for {}).".format(
                TILE_LIMIT, number_tiles
            )
        )


def validate_image_col_row(image, col, row):
    """Basic checks for columns and rows values"""
    SPLIT_LIMIT = 99

    try:
        col = int(col)
        row = int(row)
    except BaseException:
        raise ValueError("columns and rows values could not be cast to integer.")

    if col < 1 or row < 1 or col > SPLIT_LIMIT or row > SPLIT_LIMIT:
        raise ValueError(
            f"Number of columns and rows must be between 1 and"
            f"{SPLIT_LIMIT} (you asked for rows: {row} and col: {col})."
        )
    if col == 1 and row == 1:
        raise ValueError("There is nothing to divide. You asked for the entire image.")



def slice(
    image,
    number_tiles=None,
    col=None,
    row=None,
    save=True,
    DecompressionBombWarning=True,
):
    """
    Split an image into a specified number of tiles.

    Args:
       filename (str):  The filename of the image to split.
       number_tiles (int):  The number of tiles required.

    Kwargs:
       save (bool): Whether or not to save tiles to disk.
       DecompressionBombWarning (bool): Whether to suppress
       Pillow DecompressionBombWarning

    Returns:
        Tuple of :class:`Tile` instances.
    """
    if DecompressionBombWarning is False:
        Image.MAX_IMAGE_PIXELS = None

    #im = Image.open(filename)
    im = image
    im_w, im_h = im.size

    columns = 0
    rows = 0
    if number_tiles:
        validate_image(im, number_tiles)
        columns, rows = calc_columns_rows(number_tiles)
    else:
        validate_image_col_row(im, col, row)
        columns = col
        rows = row

    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1, int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    prefix="TemporalImage"
    directory=create_temp_dir()
    if save:
        save_tiles(
            tiles, prefix, directory
        )
    return tuple(tiles)

def create_temp_dir():
    TempDir = "./Temp"
    try:
       os.mkdir(TempDir)
    finally:
        return TempDir

def delete_temp_dir():
    TempDir = "./Temp"
    for files in sorted(os.listdir(TempDir)):
        os.remove(TempDir+"/"+files)
    try:
       os.rmdir(TempDir)
    finally:
        return

def save_tiles(tiles, prefix="", directory=os.getcwd(), format="png"):
    """
    Write image files to disk. Create specified folder(s) if they
       don't exist. Return list of :class:`Tile` instance.

    Args:
       tiles (list):  List, tuple or set of :class:`Tile` objects to save.
       prefix (str):  Filename prefix of saved tiles.

    Kwargs:
       directory (str):  Directory to save tiles. Created if non-existant.

    Returns:
        Tuple of :class:`Tile` instances.
    """
    for tile in tiles:
        tile.save(
            filename=tile.generate_filename(
                prefix=prefix, directory=directory, format=format
            ),
            format=format,
        )
    return tuple(tiles)


def get_image_column_row(filename):
    """Determine column and row position for filename."""
    row, column = os.path.splitext(filename)[0][-5:].split("_")
    return (int(column) - 1, int(row) - 1)


def open_images_in(directory):
    """Open all images in a directory. Return tuple of Tile instances."""

    files = [
        filename
        for filename in os.listdir(directory)
        if "_" in filename and not filename.startswith("joined")
    ]
    tiles = []
    if len(files) > 0:
        i = 0
        for file in files:
            pos = get_image_column_row(file)
            im = Image.open(os.path.join(directory, file))

            position_xy = [0, 0]
            count = 0
            for a, b in zip(pos, im.size):
                position_xy[count] = a * b
                count = count + 1
            tiles.append(
                Tile(
                    image=im,
                    position=pos,
                    number=i + 1,
                    coords=position_xy,
                    filename=file,
                )
            )
            i = i + 1
    return tiles


def get_basename(filename):
    """Strip path and extension. Return basename."""
    return os.path.splitext(os.path.basename(filename))[0]

def adjust_image_size(img,rows,cols):
    new_width=(img.size[0] // cols)*cols
    if (new_width % 2) == 1: 
        new_width=(1+(img.size[0] // cols))*cols
    else:
        new_width==(img.size[0] // cols)*cols

    new_height=(img.size[1] // rows)*rows
    if (new_height % 2) == 1: 
        new_height=(1+(img.size[1] // rows))*rows
    else:
        new_height==(img.size[1] // rows)*rows

    return img.resize([new_width, new_height])


def open_images(directory):
    """Open all images in a directory. Return tuple of Image instances."""
    return [Image.open(os.path.join(directory, file)) for file in os.listdir(directory)]


def get_columns_rows(filenames):
    """Derive number of columns and rows from filenames."""
    tiles = []
    for filename in filenames:
        row, column = os.path.splitext(filename)[0][-5:].split("_")
        tiles.append((int(row), int(column)))
    rows = [pos[0] for pos in tiles]
    columns = [pos[1] for pos in tiles]
    num_rows = max(rows)
    num_columns = max(columns)
    return (num_columns, num_rows)


def create_slice_of_substitute_for_column_joint(file,pixels=8):
    img = Image.open(file)
    img_middle=img.size[0]/2
    img = img.crop((img_middle-pixels,0,img_middle+pixels,img.size[1]))
    img.save(file)

def create_slice_of_substitute_for_row_joint(file,pixels=8):
    img = Image.open(file)
    img_middle=img.size[1]/2
    img = img.crop((0,img_middle-pixels,img.size[0],img_middle+pixels))
    img.save(file)


def substitute_image_joint_vertical_marks(img, tiles, pixels, cols):
    pixels=8
    for tile_mark in tiles:
        try:
            coord=get_image_column_row(tile_mark.filename)
            position_x=((coord[0]+1)*(int(img.size[0]/cols)))-pixels
            img.paste(tile_mark.image, [position_x,tile_mark.coords[1]])
        except IOError:
            # do nothing, blank out the image
            continue

    return img


def substitute_image_joint_horizontal_marks(img, tiles, pixels, rows):
    pixels=8
    for tile_mark in tiles:
        try:
            coord=get_image_column_row(tile_mark.filename)
            position_y=((coord[1]+1)*(int(img.size[1]/rows)))-pixels
            img.paste(tile_mark.image, [tile_mark.coords[0],position_y,])
        except IOError:
            # do nothing, blank out the image
            continue

    return img



