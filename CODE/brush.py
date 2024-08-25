## -- IMPORTS

from diffusers import FluxPipeline;
import os;
import pandas;
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

if ( argument_count >= 1 ) :

    data_file_path = GetLogicalPath( argument_array[ 1 ] );
    image_folder_path = GetLogicalPath( argument_array[ 2 ] ) if ( argument_count >= 2 ) else "";
    default_image_count = int( argument_array[ 3 ] ) if ( argument_count >= 3 ) else 1;
    default_width = int( argument_array[ 6 ] ) if ( argument_count >= 6 ) else 1024;
    default_height = int( argument_array[ 7 ] ) if ( argument_count >= 7 ) else 1024;
    default_inference_step_count = int( argument_array[ 5 ] ) if ( argument_count >= 5 ) else 4;
    default_guidance_scale = float( argument_array[ 4 ] ) if ( argument_count >= 4 ) else 0;
    default_upscaling_factor = int( argument_array[ 8 ] ) if ( argument_count >= 8 ) else 2;

    if ( data_file_path.endswith( ".csv" )
         and image_folder_path.endswith( "/" ) ) :

        print( "Loading model..." );
        model = (
            FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype = torch.bfloat16
                )
            );
        model.enable_sequential_cpu_offload();

        print( "Reading data :", data_file_path );
        data_frame = pandas.read_csv( data_file_path );
        data_frame.fillna( "", inplace = True );

        for _, row in data_frame.iterrows() :

            prompt = row[ "prompt" ];
            default_image_file_label = re.sub( r"_+", "_", re.sub( r"\W", "_", prompt ) ).strip( "_" );

            image_file_label = row[ "image_file_label" ] if ( row[ "image_file_label" ] != "" ) else default_image_file_label;
            image_count = int( row[ "image_count" ] ) if ( row[ "image_count" ] != "" ) else default_image_count;
            width = int( row[ "width" ] ) if ( row[ "width" ] != "" ) else default_width;
            height = int( row[ "height" ] ) if ( row[ "height" ] != "" ) else default_height;
            inference_step_count = int( row[ "inference_step_count" ] ) if ( row[ "inference_step_count" ] != "" ) else default_inference_step_count;
            guidance_scale = float( row[ "guidance_scale" ] ) if ( row[ "guidance_scale" ] != "" ) else default_guidance_scale;
            upscaling_factor = int( row[ "upscaling_factor" ] ) if ( row[ "upscaling_factor" ] != "" ) else default_upscaling_factor;

            print( "Processing prompt :", prompt );

            for image_index in range( image_count ) :

                if ( image_index == 0 ) :

                    image_file_path = image_folder_path + image_file_label + ".png";

                else :

                    image_file_path = image_folder_path + image_file_label + "_" + str( image_index + 1 ) + ".png";

                if ( os.path.exists( image_file_path ) ) :

                    print( "Keeping image :", image_file_path );

                else :

                    print( "Drawing image :", image_file_path );
                    seed = random.randint( 1, 999999 );
                    image = (
                        model(
                            prompt,
                            width = width,
                            height = height,
                            num_inference_steps = inference_step_count,
                            guidance_scale = guidance_scale,
                            max_sequence_length = 256,
                            generator = torch.Generator( "cpu" ).manual_seed( seed )
                            ).images[ 0 ]
                        );

                    print( "Saving image :", image_file_path );
                    image.save( image_file_path );

                    if ( upscaling_factor > 1 ) :

                        print( "Upscaling image :", image_file_path );
                        upscaled_image = image.resize( ( width * upscaling_factor, height * upscaling_factor ), 1 );
                        upscaled_image_file_path = image_file_path[ :-4 ] + ".upscaled.png";

                        print( "Saving upscaled image :", upscaled_image_file_path );
                        upscaled_image.save( upscaled_image_file_path );

        sys.exit( 0 );

print( f"*** Invalid arguments : { argument_array }" );
print( "Usage: python brush.py data.csv <guidance scale> <inference step count> <image count> IMAGE_FOLDER/" );
