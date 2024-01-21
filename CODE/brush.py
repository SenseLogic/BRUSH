## -- IMPORTS

import csv;
from diffusers import AutoPipelineForText2Image;
import os;
import random;
import re;
import torch;
import sys;

## -- FUNCTIONS

def GetLogicalPath( path ) :

    return path.replace( '\\', '/' );

## -- STATEMENTS

argument_array = sys.argv;
argument_count = len( argument_array ) - 1;

if ( argument_count == 4 ) :

    data_file_path = GetLogicalPath( argument_array[ 1 ] );
    step_count = int( argument_array[ 2 ] );
    image_count = int( argument_array[ 3 ] );
    image_folder_path = GetLogicalPath( argument_array[ 4 ] );

    if ( data_file_path.endswith( ".csv" )
         and image_folder_path.endswith( "/" ) ) :

        model_name = "stabilityai/sdxl-turbo";
        device_name ="cuda" if torch.cuda.is_available() else "cpu";

        print( f"Loading model on { device_name }..." );

        model = (
            AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype = torch.float16,
                variant = "fp16"
                )
            );

        device = torch.device( device_name );
        model.to( device );

        print( "Reading data :", data_file_path );

        with open( data_file_path, newline = "", encoding = "utf-8" ) as data_file :

            csv_reader = csv.reader( data_file );
            next( csv_reader );

            for row in csv_reader :

                image_file_label, prompt = row;

                if ( image_file_label == "" ) :

                    image_file_label = re.sub( r"_+", "_", re.sub( r"\W", "_", prompt ) ).strip( "_" );

                print( "Processing prompt :", prompt );

                for image_index in range( image_count ) :

                    image_file_path = image_folder_path + image_file_label + "_" + str( image_index + 1 ) + ".png";

                    if ( os.path.exists( image_file_path ) ) :

                        print( "Keeping image :", image_file_path );

                    else :

                        print( "Drawing image :", image_file_path );

                        seed = random.randint( 1, 999999 );

                        image = (
                            model(
                                prompt,
                                num_inference_steps = step_count,
                                guidance_scale = 0.0,
                                generator = torch.manual_seed( seed )
                                ).images[ 0 ]
                            );

                        print( "Saving image :", image_file_path );
                        image.save( image_file_path );

                        upscaled_image = image.resize( ( 1024, 1024 ), 1 );
                        upscaled_image_file_path = image_file_path[ :-4 ] + "_upscaled.png";

                        print( "Saving upscaled image :", upscaled_image_file_path );
                        upscaled_image.save( upscaled_image_file_path );

        sys.exit( 0 );

print( f"*** Invalid arguments : { argument_array }" );
print( "Usage: python brush.py data.csv <step count> <image count> IMAGE_FOLDER/" );
