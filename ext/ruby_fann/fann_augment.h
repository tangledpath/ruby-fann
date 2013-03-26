#include "ruby.h"
#include "ruby_compat.h"

FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train_from_rb_ary2(
	unsigned int num_data,
	unsigned int num_input,
  unsigned int num_output)
{
	return 0;
}

/*
 * Copied from fann_create_train_from_callback/file & modified to ease 
 * allocating from ruby arrays:
 */
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train_from_rb_ary(
	VALUE inputs, 
	VALUE outputs 
)
{
    unsigned int i, j;
    fann_type *data_input, *data_output;
    struct fann_train_data *data = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));
    unsigned int num_input = RARRAY_LEN(RARRAY_PTR(inputs)[0]);
    unsigned int num_output =RARRAY_LEN(RARRAY_PTR(outputs)[0]);
		unsigned int num_data = RARRAY_LEN(inputs);
		
    if(data == NULL) {
        fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }

    fann_init_error_data((struct fann_error *) data);

    data->num_data     = num_data;
    data->num_input    = num_input;
    data->num_output = num_output;

    data->input = (fann_type **) calloc(num_data, sizeof(fann_type *));
    if(data->input == NULL)
    {
        fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    data->output = (fann_type **) calloc(num_data, sizeof(fann_type *));
    if(data->output == NULL)
    {
        fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    data_input = (fann_type *) calloc(num_input * num_data, sizeof(fann_type));
    if(data_input == NULL)
    {
        fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    data_output = (fann_type *) calloc(num_output * num_data, sizeof(fann_type));
    if(data_output == NULL)
    {
        fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

		VALUE inputs_i, outputs_i;
		for(i = 0; i != num_data; i++)
		{
			data->input[i] = data_input;
			data_input += num_input;

			inputs_i = RARRAY_PTR(inputs)[i];
			outputs_i = RARRAY_PTR(outputs)[i];
			
			if(RARRAY_LEN(inputs_i) != num_input) 
			{
				rb_raise (
					rb_eRuntimeError, 
					"Number of inputs at [%d] is inconsistent: (%d != %d)", 
					i, RARRAY_LEN(inputs_i), num_input);
			}
			
			if(RARRAY_LEN(outputs_i) != num_output) 
			{
				rb_raise (
					rb_eRuntimeError, 
					"Number of outputs at [%d] is inconsistent: (%d != %d)", 
					i, RARRAY_LEN(outputs_i), num_output);
			}
			
			
			for(j = 0; j != num_input; j++)
			{
				data->input[i][j]=NUM2DBL(RARRAY_PTR(inputs_i)[j]);
			}

			data->output[i] = data_output;
			data_output += num_output;
			
			for(j = 0; j != num_output; j++)
			{
				data->output[i][j]=NUM2DBL(RARRAY_PTR(outputs_i)[j]);
			}
		}
		
    return data;
}