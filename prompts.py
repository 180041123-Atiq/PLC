


def gen_layout_prompt(index):

    # Open the file
    layout = ''
    
    with open(f"webcode2m_plc/layout/{index}.txt", "r") as f:
        
        for line in f:
            # Remove any trailing newline or spaces
            line = line.strip()
            layout+= line
            layout += ' '
            if line == '3':
                break
    # print(layout)

    pp = "You are an assistant that turns a screenshot of a webpage into clean HTML with inline CSS. " +\
    "A layout sequence containing 4, 0, 1, 2, and 3, is given to show how the elements are arranged, where\n" +\
    "- 4 : Start of the sequence,\n" +\
    "- 0 : Start an HTML tag whose children go side by side (horizontal),\n" +\
    "- 1 : Start an HTML tag whose children go on top of each other (vertical),\n" +\
    "- 2 : Close the most recent HTML tag,\n" +\
    "- 3 : End of the sequence\n" +\
    "Use the following layout sequence to guide your HTML + inline CSS code generation. " +\
    "Output only HTML + inline CSS in a single markdown code block. " +\
    "If the layout has mistakes, make the HTML correct as best as you can.\n" +\
    "Layout Sequence: " +\
    f"{layout}"

    return pp

if __name__ == '__main__':
    print(gen_layout_prompt(106))