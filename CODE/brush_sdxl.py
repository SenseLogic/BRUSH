## -- IMPORTS

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler;
from huggingface_hub import hf_hub_download;
from safetensors.torch import load_file;
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
    default_width = int( argument_array[ 3 ] ) if ( argument_count >= 3 ) else 1024;
    default_height = int( argument_array[ 4 ] ) if ( argument_count >= 4 ) else 1024;
    default_image_count = int( argument_array[ 5 ] ) if ( argument_count >= 5 ) else 1;
    default_inference_step_count = int( argument_array[ 6 ] ) if ( argument_count >= 6 ) else 4;
    default_guidance_scale = float( argument_array[ 7 ] ) if ( argument_count >= 7 ) else 1.8;

    if ( data_file_path.endswith( ".csv" )
         and image_folder_path.endswith( "/" ) ) :

        print( "Loading model..." );
        base_model = "stabilityai/stable-diffusion-xl-base-1.0";
        lightning_model = "ByteDance/SDXL-Lightning";

        default_steps = default_inference_step_count;
        if ( default_steps <= 2 ) :
            lightning_checkpoint_name = "sdxl_lightning_2step_unet.safetensors";
        elif ( default_steps <= 4 ) :
            lightning_checkpoint_name = "sdxl_lightning_4step_unet.safetensors";
        else :
            lightning_checkpoint_name = "sdxl_lightning_8step_unet.safetensors";

        unet_config = UNet2DConditionModel.load_config( base_model, subfolder = "unet" );
        unet = UNet2DConditionModel.from_config( unet_config ).to( "cuda", torch.float16 );
        unet.load_state_dict( load_file( hf_hub_download( lightning_model, lightning_checkpoint_name ), device = "cuda" ) );

        pipeline = (
            StableDiffusionXLPipeline.from_pretrained(
                base_model,
                unet = unet,
                torch_dtype = torch.float16,
                variant = "fp16"
                ).to( "cuda" )
            );
        pipeline.scheduler = EulerDiscreteScheduler.from_config( pipeline.scheduler.config, timestep_spacing = "trailing" );
        pipeline.enable_attention_slicing();
        # pipeline.enable_vae_slicing();
        # pipeline.enable_model_cpu_offload();

        print( "Reading data :", data_file_path );
        data_frame = pandas.read_csv( data_file_path, dtype = str );
        data_frame = data_frame.fillna( "" );
        data_frame = data_frame.replace( "nan", "" );
        data_frame = data_frame.replace( "None", "" );

        for _, row in data_frame.iterrows() :

            prompt = row[ "prompt" ];
            default_image_file_label = re.sub( r"_+", "_", re.sub( r"\W", "_", prompt ) ).strip( "_" );

            image_file_label = row[ "image_file_label" ] if ( row[ "image_file_label" ] != "" ) else default_image_file_label;
            image_count = int( row[ "image_count" ] ) if ( row[ "image_count" ] != "" ) else default_image_count;
            width = int( row[ "width" ] ) if ( row[ "width" ] != "" ) else default_width;
            height = int( row[ "height" ] ) if ( row[ "height" ] != "" ) else default_height;
            inference_step_count = int( row[ "inference_step_count" ] ) if ( row[ "inference_step_count" ] != "" ) else default_inference_step_count;
            guidance_scale = float( row[ "guidance_scale" ] ) if ( row[ "guidance_scale" ] != "" ) else default_guidance_scale;

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
                        pipeline(
                            prompt,
                            width = width,
                            height = height,
                            num_inference_steps = inference_step_count,
                            guidance_scale = guidance_scale,
                            generator = torch.Generator( "cpu" ).manual_seed( seed )
                            ).images[ 0 ]
                        );

                    print( "Saving image :", image_file_path );
                    image.save( image_file_path );

        sys.exit( 0 );

print( f"*** Invalid arguments : { argument_array }" );
print( "Usage: python brush.py data.csv <IMAGE_FOLDER/|> <image width|1024> <image height|1024> <image count|1> <inference step count|4> <guidance scale|1.8>" );
