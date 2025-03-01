from PIL import Image
from io import BytesIO
from vllm import LLM
from vllm.sampling_params import SamplingParams
import base64


def MLLM_model_sampling_loader(n=1,temperature=0,model_name = "mistralai/Pixtral-12B-2409"):
    llm = LLM(model=model_name, tokenizer_mode="mistral",limit_mm_per_prompt={"image": 8},enforce_eager=True,max_model_len=32648)
    sampling_params = SamplingParams(max_tokens=8192,n=n,temperature =temperature)
    return llm.chat,sampling_params



def generate_system_content(template_type):

    template_dict = {
    'prompt_processing': {
    "role": "system",
    "content": 
        """
            Please process the following prompt intended for an image generation model. Your task is to analyze and restructure the prompt into four *DISTINCT* parts without introducing or inferring any information that is not explicitly present in the original prompt. 
            The output shouldremain entirely faithful to the original wording, ensuring complete and accurate preservation of relationships and details.
            
            1. **Image Content**: 
            Extract and maintain full descriptions of the objects, characters, scenery, and their relationships as explicitly stated in the prompt. This section should capture the essence of what is depicted rather than reducing it to isolated keywords.

            2. **Magic Words**:
            Identify and extract terms or phrases commonly used to enhance image generation quality, such as platform references (e.g., "ArtStation," "Pixiv"), artist names (e.g., "Greg Rutkowski"), or general quality-enhancing descriptors like "4K," "highly detailed," etc.

            3. **Artistic Style**:
            Extract explicit terms describing the visual art style stated in the prompt, such as "watercolor," "cyberpunk," or other recognized stylistic descriptors. Do not include terms that are classified as magic words in this category.

            4. **Visual Properties**:
            Extract descriptions of the image’s visual attributes and composition, such as lighting, camera angles, depth of field, texture, atmosphere, and other  physical properties influencing the scene’s appearance. Do not include terms that are classified as magic words in this category.
        
            Structure the output in the following JSON format, ensuring no additional content or inferred details are included:
            {
                "image_content": "Full description of what is depicted in the image",
                "magic_words": ["list of explicit magic words"],
                "artistic_style": ["list of explicit style-related terms"],
                "visual_properties": ["list of explicit visual attribute-related terms"]
            }
        """
        },
    'prompt_NSFW_filtering': {
    "role": "system",
    "content": 
        """
            Task: Analyze a text-to-image generation prompt and determine if it contains any NSFW (Not Safe For Work) content, including explicit or inappropriate implications that may not be immediately obvious. 
            **Guidelines**:
            1. **Consider Implicit Meaning**:
            Pay attention not only to explicit terms but also to phrases or descriptions that may imply NSFW content, even if the language appears neutral.
            
            2. **Context Matters**:  
            Analyze the overall context of the prompt to detect hidden or suggestive meanings.

            3. **Binary Output**:
            Classify as either {"NSFW Detected": true} or {"NSFW Detected": false}.

            4. **Prioritize Safety**:
            When in doubt, lean towards marking as NSFW if there is any reasonable implication of inappropriate content.
        
            Output your response in the following JSON format:
            {
            "NSFW Detected": true
            }
            and don't output anything else.
        """
        },
    
    'prompt_classification': {
    "role": "system",
    "content": 
        """
        Task: Please classify the given image generation prompt into one of the following categories based on its content. Focus only on the main subject of the prompt, ignoring secondary or less important elements.

        **Abstract & Artistic**: Includes non-representational graphics, decorative patterns, textures, stylized illustrations, and cartoon-like imagery.
            
        **Animals & Plants**: Covers real-world animals and plants, as well as their mythical and fictional counterparts, such as legendary creatures and imaginary flora.
            
        **Characters**: Encompasses real or fictional human figures, as well as humanoid sci-fi or fantasy beings like elves, aliens, or mechas.
            
        **Objects & Food**: Consists of man-made items, including furniture, tools, technological devices, and mechanical transportation (e.g., cars, ships, airplanes). Also includes edible items such as ingredients, beverages, and complete dishes or meals.
            
        **Scenes**: Refers to both indoor environments such as rooms, offices, or factories, and outdoor settings like natural landscapes, city streets, or parks.

        Output your response in the following JSON format:

        {
            "category": "Selected category name"
        }

        and don’t output anything else.
        """
        },  
    
        
    'captioning_coco_image': {
    "role": "system",
    "content": 
        """
        Task: I will give you an image. Describe the image **precisely** but don't mention anything that have nothing to do with the main content of the image. 

        Output your response in the following JSON format:
        {
        "prompt": ...
        }
        and don’t output anything else.
        """
        },  
    
    'captioning_aes_image': {
    "role": "system",
    "content": 
        """
        Task: I will give you an image. Analyze the provided image and describe its content concisely. Additionally, provide a brief description of the artistic style and notable visual properties of the image. 

        Output your response in the following JSON format:
        {
            "image_content": "Brief description of the image content.",
            "artistic_style": "Description of the artistic style.",
            "visual_properties": "Key visual features such as color palette,lighting, composition, texture, etc."
        }
        and don’t output anything else.
        """
        },  
        
    'annotation_alignment': {
    "role": "system",
    "content": 
        """
        Task:

        You are a professional digital artist. Your task is to evaluate how well AI-generated images **align** the given text prompt ("Input").

        **Evaluation Rules:**
        
        - **Alignment**: Score how well the image reflects the details and intent of the prompt.
         
        -  The images are **independent**, and should be evaluated separately and step by step.
        
        -**Alignment Rating Scale (0–100):**           
            **90–100**: Perfect match. The image fully aligns with all details and intent of the prompt.
            **70–89**: Strong match. The image aligns with most of the prompt but has minor inaccuracies or missing details.
            **50–69**: Partial match. The image aligns with the general idea but misses or misrepresents key elements.
            **30–49**: Weak match. The image barely aligns with the prompt, with many critical details missing or incorrect.
            **0–29**: No match. The image does not align with the prompt at all.
            


        **Output Format**:

        Provide your evaluation as a JSON object. Include a separate alignment score for each image based on its prompt. You have **NUM** images to be evaluated, so do not miss any of them. Use the following format:

        {
            "image_1": {"alignment": ...},
            "image_2": {"alignment": ...},
            "image_3": {"alignment": ...},
            ...
        }

        Do not include reasoning, explanations, or any additional text.
        """
        },  
    
    
    'annotation_aesthetic': {
    "role": "system",
    "content": 
        """
        Task:

        You are a professional digital artist. Your task is to evaluate the **aesthetic** quality of AI-generated images based on their visual appeal and artistic merit as described in the given text prompt ("Input"). 
        Focus on aspects like clarity, lighting, color harmony, and composition.

        **Evaluation Rules:**
            - **Aesthetic**: Rate the visual and artistic quality of the image based on the following factors:
                - **Clarity**: Is the image sharp and free from blurriness?
                - **Lighting**: Is the exposure balanced and effective?
                - **Colors**: Are the colors vibrant and harmonious?
                - **Composition**: Does the image have a clear focal point and thoughtful design?
            - The images are **independent** and should be evaluated separately, step by step.
        
        **Aesthetic Rating Scale (0–100):**
            - **90–100**: **Excellent**. The image is visually stunning with exceptional clarity, perfect lighting, rich colors, and a masterful composition that evokes emotion.
            - **70–89**: **Good**. The image is sharp, well-exposed, with vibrant colors and a thoughtful composition that has minor areas for improvement.
            - **50–69**: **Fair**. The image is in focus with adequate lighting and decent composition but lacks creativity or artistic impact.
            - **30–49**: **Poor**. The image has noticeable issues like blur, poor lighting, washed-out colors, or awkward composition.
            - **0–29**: **Bad**. The image is extremely blurry, poorly lit, noisy, with chaotic composition and indiscernible subjects.
                 
        **Output Format**:

        Provide your evaluation as a JSON object. Include a separate alignment score for each image based on its prompt. You have **NUM** images to be evaluated, so do not miss any of them. Use the following format:

        {
            "image_1": {"alignment": ...},
            "image_2": {"alignment": ...},
            "image_3": {"alignment": ...},
            ...
        }

        Do not include reasoning, explanations, or any additional text.
        """
        },      
    
    'annotation_fidelity': {
    "role": "system",
    "content": 
        """
        Task:

        You are a professional digital artist. Your task is to evaluate the **fidelity} of AI-generated images based on their adherence to expected shapes and characteristics as described in the given text prompt ("Input"). 
        A higher fidelity score denotes that the image remains true to the expected shape and characteristics of the object, avoiding haphazard or incorrect features.

        **Evaluation Rules:**

            - **Fidelity**: Rate how accurate the image is in terms of shapes, anatomy, and other characteristics that should logically match the description in the prompt.
            - The images are **independent** and should be evaluated separately, step by step.
            - Use the following examples to understand low-fidelity issues:
                    - **"Spider-Man"** should only have two legs. If an image has extra legs, it is incorrect.
                    - **"Unicorn"** should have one horn. Multiple horns would be incorrect.
                    - A human hand should have **five fingers**. More or fewer fingers are incorrect.
    
        **Fidelity Rating Scale (0–100):**

            - **90–100**: No errors in shapes or characteristics. The objects are technically accurate and look natural.
            - **70–89**: Minor errors in shape or characteristics (e.g., slightly incorrect details) but overall acceptable and well-coordinated.
            - **50–69**: Noticeable errors in shape or characteristics, but the image remains somewhat coherent.
            - **30–49**: Major errors in shape or characteristics that disrupt the overall impression of the image.
            - **0–29**: Severe errors that make the objects unrealistic or nonsensical.
                 
        **Output Format**:

        Provide your evaluation as a JSON object. Include a separate alignment score for each image based on its prompt. You have **NUM** images to be evaluated, so do not miss any of them. Use the following format:

        {
            "image_1": {"alignment": ...},
            "image_2": {"alignment": ...},
            "image_3": {"alignment": ...},
            ...
        }

        Do not include reasoning, explanations, or any additional text.
        """
        },  
        

}
     
    return template_dict[template_type]       
    
    
    
def image_to_base64(image, format="JPEG"):
    def _process_image(image):
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image) 
        return image  

    image = _process_image(image)
    if image is None:
        raise ValueError("Unsupported image format")

    buffered = BytesIO()
    if image.mode in ("RGBA", "P"): 
        image = image.convert("RGB")
    image.save(buffered, format=format)

    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

        
def generate_image_prompt_request(prompt, image_list, template_type):         
    system_content = generate_system_content(template_type)              
    input_text = "## Input: {INSERT PROMPT HERE}\n".replace("{INSERT PROMPT HERE}", prompt)
    system_content['content'].replace('NUM', str(len(image_list)))
    messages = []
    messages.append(system_content)
    content = [
        {
            "type": "text",
            "text": input_text
        }]
    for idx,image in enumerate(image_list) :
        content.append({"type": "text", "text": f"\nImage {idx+1}:\n"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image)}"}})
    messages.append({
    "role": "user",
    "content": content})
    
    return messages  

def generate_prompt_request(prompt, template_type):         
    system_content = generate_system_content(template_type)              
  
    messages = []
    messages.append(system_content)
    messages.append({
    "role": "user",
    "content": prompt})
    
    return messages  
    
    
def generate_image_request(image, template_type):         
    system_content = generate_system_content(template_type)              
    messages = []
    messages.append(system_content)
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "\nImage :\n"},
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image)}"}},   
        ]
    })
    
    return messages     


def generate_request(template_type,image=None,prompt=None):   
    if template_type.startswith('prompt'):
        messages = generate_prompt_request(prompt,template_type) 
    elif template_type.startswith('captioning'):
        messages = generate_image_request(image,template_type) 
    elif template_type.startswith('annotation'):
        messages = generate_image_prompt_request(prompt, image, template_type) 
    return messages  