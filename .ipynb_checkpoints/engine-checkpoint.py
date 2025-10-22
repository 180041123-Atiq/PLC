import glob
import argparse
import os
import random
from PIL import Image
import shutil

from qwen import gen_qwen_output
from processingData import TestingLayouts
from output_qwen import first_qwen_output
from prompts import gen_layout_prompt

def save_output_html(mllm_output,path):

    start_html = mllm_output.find('<')
    end_html = mllm_output.rfind('>')

    # Extract the substring
    if start_html != -1 and end_html != -1 and start_html < end_html:
        result_output = mllm_output[start_html:end_html+1]  # include '>'
        # print(result_output)
        TestingLayouts.save_str_html(result_output,path)
    else:
        print("No valid '<...>' found")

def gen_args():
    # Create parser
    parser = argparse.ArgumentParser(description="Engine of PLC")

    # Add arguments
    parser.add_argument(
        "--dummy-run",        # argument name
        type=bool,        # type of argument
        default=False, # default value
        help="Dummy run is for only testing"
    )

    parser.add_argument(
        "--save-dir",        # argument name
        type=str,        # type of argument
        default='outputs', # default value
        help="path to save predictions and model and result"
    )

    parser.add_argument(
        "--data-dir",        # argument name
        type=str,        # type of argument
        default='webcode2m_plc', # default value
        help="path to save predictions and model and result"
    )

    parser.add_argument(
        "--mllm",        # argument name
        type=str,        # type of argument
        default='qwen', # default value
        help="mllm used to gen HTML"
    )

    # Parse arguments
    return parser.parse_args()

if __name__ == '__main__':

    args = gen_args()

    mllm_outputs = []

    predictions_dir = os.path.join(args.save_dir,'predictions')
    if os.path.exists(predictions_dir):
        shutil.rmtree(predictions_dir)
    os.makedirs(predictions_dir) 
    
    if args.mllm == 'qwen':
        gen_mllm_output = gen_qwen_output

    if args.dummy_run:
        mllm_outputs.append(first_qwen_output)
    else:
        html_list = glob.glob(os.path.join(args.data_dir,"code/*.html"))
        html_image_list = glob.glob(os.path.join(args.data_dir,"image/*.png"))
        html_layout_list = glob.glob(os.path.join(args.data_dir,"layout/*.txt"))
        
        sampled_list = random.sample([ee for ee in range(1000)], 100)

        # Now loop through the sampled ones
        for ii,index in enumerate(sampled_list):
            html_image = Image.open(html_image_list[index])
            html_layout = gen_layout_prompt(index)
            print(f"{index} is getting generated")
            mllm_outputs.append(gen_mllm_output(html_image,html_layout)) 
        
            save_output_html(
                mllm_output=mllm_outputs[-1],
                path=os.path.join(predictions_dir,f'{index}.html')
            )
            break

     

    # for i in range(len(mllm_outputs)):
    #     save_output_html(
    #         mllm_output=mllm_outputs[i],
    #         path=os.path.join(predictions_dir,f'{i}.html')
    #     )

    