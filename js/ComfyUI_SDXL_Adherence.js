import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "comfyui_sdxl_adherence", // Will be replaced at creation

    async setup() {
        // This function runs when the extension is loaded
        console.log("[] Extension loaded.");
        // Example: Add a custom button to the UI (optional)
        // const btn = document.createElement('button');
        // btn.textContent = 'Click me!';
        // btn.onclick = () => alert('Button clicked!');
        // document.body.appendChild(btn);
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // This function runs before the node is registered in the UI
        if (nodeData.name === "ComfyUI_SDXL_Adherence") {
            // Set a custom description
            nodeData.description = "This is a custom node for ComfyUI. It demonstrates various input types and options.";

            // Customize the node's display
            nodeData.displayOptions = {
                // Show the input_text in the node's title if available
                title: function(node, inputData) {
                    const inputText = inputData.input_text?.[0];
                    return inputText ? `: ${inputText}` : "ComfyUI_SDXL_Adherence";
                },
                // Add a custom toolbar button
                toolbar: [
                    {
                        name: "Info",
                        action: function(node, app) {
                            app.ui.dialog.show(
                                " Info",
                                `This node processes various inputs and returns transformed outputs.
                                \n\nInputs: text, number, boolean, choice, and optional text.
                                \nOutputs: transformed versions of each input.`
                            );
                        }
                    }
                ]
            };
        }
    }
});

// For advanced usage, see the ComfyUI extension API documentation and source code.
