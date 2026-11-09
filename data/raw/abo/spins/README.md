# Amazon Berkeley Objects (c) by Amazon.com

[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)
is a collection of product listings with multilingual metadata, catalog
imagery, high-quality 3d models with materials and parts, and benchmarks derived
from that data.

## License

This work is licensed under the Creative Commons Attribution 4.0 International
Public License. To obtain a copy of the full license, see LICENSE-CC-BY-4.0.txt,
visit [CreativeCommons.org](https://creativecommons.org/licenses/by/4.0/)
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Under the following terms:

  * Attribution — You must give appropriate credit, provide a link to the
    license, and indicate if changes were made. You may do so in any reasonable
    manner, but not in any way that suggests the licensor endorses you or your
    use.

  * No additional restrictions — You may not apply legal terms or technological
    measures that legally restrict others from doing anything the license
    permits.
    
## Attribution

Credit for the data, including all images and 3d models, must be given to:

> Amazon.com

Credit for building the dataset, archives and benchmark sets must be given to:

> Matthieu Guillaumin (Amazon.com), Thomas Dideriksen (Amazon.com),
> Kenan Deng (Amazon.com), Himanshu Arora (Amazon.com),
> Jasmine Collins (UC Berkeley) and Jitendra Malik (UC Berkeley)

## Description

The `spins/` directory and `abo-spins.tar` archives are made of the following
files:

  * `LICENSE-CC-BY-4.0.txt` - The License file. You must read, agree and
    comply to the License before using the Amazon Berkeley Objects data.

  * `spins/metadata/spins.csv.gz` - Spin / 360º-View image metadata. This file
    is a gzip-compressed comma-separated value (CSV) file with the following
    fields: `spin_id`, `azimuth`, `image_id`, `height`, `width`, and `path`.
    - `spin_id`: a unique identifier for the image sequence.
    - `azimuth`: an integer between 0 and 71, representing the index in the spin
      sequence and the azimuth of the camera (in steps of 5º).
    - `image_id`: this id uniquely refers to an image. This is can be used
      to retrieve the image data using the template:
            `https://m.media-amazon.com/image/I/<image_id>.jpg` [^1].
    - `height` and `width`: respectively, the height and width of the image.
    - `path`: the location of the image file relative to the `spins/original/`
      directory. The extension is `jpg` for all the images. The `path` is build
      from the `spin_id` and `azimuth` using the template
      `<spin_id>_<azimuth:02d>.jpg` and the first two characters of `spin_id`
      are used to build a file hierarchy and reduce the number of files in a
      single directory.

    Below are are first 10 lines of `spins/metadata/spins.csv.gz`:
```
spin_id,azimuth,image_id,height,width,path
61c91265,0,41wqHws7a6L,248,1075,61/61c91265/61c91265_00.jpg
61c91265,1,41++eZZHP9L,248,1075,61/61c91265/61c91265_01.jpg
61c91265,2,41YF86LhGDL,248,1075,61/61c91265/61c91265_02.jpg
61c91265,3,41I5Zz-kbAL,248,1075,61/61c91265/61c91265_03.jpg
61c91265,4,41lAQM2Ys5L,248,1075,61/61c91265/61c91265_04.jpg
61c91265,5,41OJT+p8JgL,248,1075,61/61c91265/61c91265_05.jpg
61c91265,6,412kYqOnqHL,248,1075,61/61c91265/61c91265_06.jpg
61c91265,7,41rgUZ0NuFL,248,1075,61/61c91265/61c91265_07.jpg
61c91265,8,41PJ4ks-cWL,248,1075,61/61c91265/61c91265_08.jpg
```

  * `spins/original/<path>` - Spin / 360º-View image files. Each file
    corresponds to one row in `spins/metadata/spins.csv.gz`, named by the value
    of the `path` column.
  
## Footnotes

[^1]: Importantly, there is no guarantee that these URLs will remain unchanged
and available on the long term, we thus recommend using the images provided in
the archives instead.
