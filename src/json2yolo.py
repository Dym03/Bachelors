# Source https://www.kaggle.com/code/ahmedtowsiftahmid/convert-mapillary-traffic-sign-annotations-to-yolo/notebook?scriptVersionId=117351543

import json
import pathlib
import pandas as pd

def from_json(json_file):
    """Create a flat dataframe from a JSON file."""

    json_file = pathlib.Path(json_file)  # In case the file path is passed as a string

    # Load json as a regular python dict
    data = json.loads(json_file.read_bytes())

    # pandas.json_normalize drops items whose record_path points to an empty list
    if not data['objects']:
        data['objects'] = [{}]  # Now the list contains an empty dict (record)

    # Create a dataframe where each line correspons to one annotation
    df = pd.json_normalize(data, 
                           record_path=['objects'],
                           meta=['ispano', 'width', 'height'])

    # Add file name (without extension) as a column and return resulting dataframe
    return df.assign(image=json_file.stem)

def to_yolo(df):
    """Convert annotations in dataframe to YOLO format."""

    new_df = pd.concat((
        df['image'],                                            # image file name
        df['label'].cat.codes,                                  # class id (number)
        df[['xmin', 'xmax']].mean(axis=1),                      # centroid x = (xmin+xmax)/2
        df[['ymin', 'ymax']].mean(axis=1),                      # centroid y = (ymin+ymax)/2
        df[['xmax', 'ymax']] - df[['xmin', 'ymin']].to_numpy()  # box width and height
    ), axis=1)

    new_df.columns = ['image', 'class', 'xc', 'yc', 'w', 'h']

    # Normalize calculated values to the image's width and height
    new_df[['xc', 'w']] = new_df[['xc', 'w']].div(df['width'], axis=0)
    new_df[['yc', 'h']] = new_df[['yc', 'h']].div(df['height'], axis=0)

    return new_df

if __name__ == '__main__':

    input_dir = pathlib.Path('datasets/Mapillary/annotations')
    output_dir = pathlib.Path('datasets/Mapillary/annotations_yolo')

    df = pd.concat(map(from_json, input_dir.glob('*.json')), ignore_index=True)

    # Converting column 'label' to type pandas.CategoricalDtype
    # makes it easier to get a numerical representation for each label
    df = df.astype({'label': 'category'})

    # Remove prefixes added by pandas.json_normalize when flattening the original json object
    for prefix in ('bbox.', 'properties.'):
        # df.columns = [c.removeprefix(prefix) for c in df.columns]  # removeprefix: Python 3.9+ only
        df.columns = [c[len(prefix):] if c.startswith(prefix) else c for c in df.columns]

    with open(output_dir / 'classes.txt', 'w') as f:
        f.write('\n'.join(df['label'].cat.categories))

    df_yolo = to_yolo(df)

    for image_name, group in df_yolo.groupby('image'):
        path = output_dir / f'{image_name}.txt'

        # If there are annotations for this image, dump them into the file
        if (group['class'] != -1).any():
            group.drop('image', axis=1).to_csv(path, sep=' ', header=False, index=False)
        else:
            path.touch()  # Create empty file

    print(len(list(output_dir.glob('*.txt'))), 'txt files in output directory')
