import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules import processing
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from modules.generation_parameters_copypaste import parse_generation_parameters
from modules.extras import run_pnginfo

class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Override from file PNGInfo"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        wantPrompt = gr.Checkbox(False, label="Override Prompt")
        wantNegative= gr.Checkbox(False, label="Override Negative Prompt")
        wantSeed = gr.Checkbox(False, label="Override Seed")
        wantSteps = gr.Checkbox(False, label="Override Steps")
        wantCFGScale = gr.Checkbox(False, label="Override CFG scale")
        wantSize = gr.Checkbox(False, label="Override Width and Height")
        wantdenoise = gr.Checkbox(False, label="Override Denoising strength")
        return [wantPrompt, wantNegative, wantSeed,wantSteps,wantCFGScale,wantSize,wantdenoise]

  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, wantPrompt, wantNegative, wantSeed,wantSteps,wantCFGScale,wantSize,wantdenoise):
        
        #proc = process_images(p)
        mytext = run_pnginfo(p.init_images[0])[1]
        deets = parse_generation_parameters(mytext)
        if wantPrompt and 'Prompt' in deets:
            p.prompt = deets['Prompt']
        if wantNegative and 'Negative prompt' in deets:
            p.negative_prompt = deets['Negative prompt']
        if wantSeed and 'Seed' in deets:
            p.seed = float(deets['Seed'])
        if wantSteps and 'Steps' in deets:
            p.steps = int(deets['Steps'])
        if wantCFGScale  and 'Seed' in deets:
            p.cfg_scale = int(deets['CFG scale'])                        
        if wantSize and 'Size-1' in deets:
            p.width = int(deets['Size-1'])
        if wantSize and 'Size-2' in deets:
            p.height = int(deets['Size-2'])
        if wantdenoise and 'Denoising strength' in deets:
            p.denoising_strength = float(deets['Denoising strength'])
        
        processing.fix_seed(p)


        return process_images(p)