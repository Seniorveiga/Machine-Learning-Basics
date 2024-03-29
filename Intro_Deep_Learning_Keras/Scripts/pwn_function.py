import numpy as np

def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights["node_0"]).sum()
    node_0_output = node_0_input

    # Calculate node 1 value
    node_1_input = (input_data_row * weights["node_1"]).sum()
    node_1_output = node_1_input

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights["output"]).sum()
    model_output = input_to_final_layer
    
    # Return model output
    return(model_output)

def predict_with_network_one_step(input_data_row, weights):

    # Calculate node 0 value
    node_input = (input_data_row * weights).sum()
    node_output = node_input
    
    # Return model output
    return(node_output)