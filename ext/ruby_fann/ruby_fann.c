#include "ruby.h"
#include "ruby_compat.h"
#include "doublefann.h"
#include "fann_data.h"
#include "fann_augment.h"

static VALUE m_rb_fann_module;
static VALUE m_rb_fann_standard_class;
static VALUE m_rb_fann_shortcut_class;
static VALUE m_rb_fann_train_data_class;

#define RETURN_FANN_INT(fn) \
struct fann* f; \
Data_Get_Struct (self, struct fann, f); \
return INT2NUM(fn(f)); 

#define SET_FANN_INT(attr_name, fann_fn) \
Check_Type(attr_name, T_FIXNUM); \
struct fann* f; \
Data_Get_Struct(self, struct fann, f); \
fann_fn(f, NUM2INT(attr_name)); \
return 0;

#define RETURN_FANN_UINT(fn) \
struct fann* f; \
Data_Get_Struct (self, struct fann, f); \
return UINT2NUM(fn(f)); 

#define SET_FANN_UINT(attr_name, fann_fn) \
Check_Type(attr_name, T_FIXNUM); \
struct fann* f; \
Data_Get_Struct(self, struct fann, f); \
fann_fn(f, NUM2UINT(attr_name)); \
return 0;

// Converts float return values to a double with same precision, avoids floating point errors.
#define RETURN_FANN_FLT(fn) \
struct fann* f; \
Data_Get_Struct (self, struct fann, f); \
char buffy[20]; \
sprintf(buffy, "%0.6g", fn(f)); \
return rb_float_new(atof(buffy)); 

#define SET_FANN_FLT(attr_name, fann_fn) \
Check_Type(attr_name, T_FLOAT); \
struct fann* f; \
Data_Get_Struct(self, struct fann, f); \
fann_fn(f, NUM2DBL(attr_name)); \
return self;

#define RETURN_FANN_DBL(fn) \
struct fann* f; \
Data_Get_Struct (self, struct fann, f); \
return rb_float_new(fn(f)); 

#define SET_FANN_DBL SET_FANN_FLT

// Convert ruby symbol to corresponding FANN enum type for activation function:
enum fann_activationfunc_enum sym_to_activation_function(VALUE activation_func)
{
    ID id=SYM2ID(activation_func);
    enum fann_activationfunc_enum activation_function;
    if(id==rb_intern("linear")) {
        activation_function = FANN_LINEAR;  
    }   else if(id==rb_intern("threshold")) {
        activation_function = FANN_THRESHOLD;   
    }   else if(id==rb_intern("threshold_symmetric")) {
        activation_function = FANN_THRESHOLD_SYMMETRIC; 
    }   else if(id==rb_intern("sigmoid")) {
        activation_function = FANN_SIGMOID; 
    }   else if(id==rb_intern("sigmoid_stepwise")) {
        activation_function = FANN_SIGMOID_STEPWISE;    
    }   else if(id==rb_intern("sigmoid_symmetric")) {
        activation_function = FANN_SIGMOID_SYMMETRIC;   
    }   else if(id==rb_intern("sigmoid_symmetric_stepwise")) {
        activation_function = FANN_SIGMOID_SYMMETRIC_STEPWISE;  
    }   else if(id==rb_intern("gaussian")) {
        activation_function = FANN_GAUSSIAN;    
    }   else if(id==rb_intern("gaussian_symmetric")) {
        activation_function = FANN_GAUSSIAN_SYMMETRIC;  
    }   else if(id==rb_intern("gaussian_stepwise")) {
        activation_function = FANN_GAUSSIAN_STEPWISE;   
    }   else if(id==rb_intern("elliot")) {
        activation_function = FANN_ELLIOT;  
    }   else if(id==rb_intern("elliot_symmetric")) {
        activation_function = FANN_ELLIOT_SYMMETRIC;    
    }   else if(id==rb_intern("linear_piece")) {
        activation_function = FANN_LINEAR_PIECE;    
    }   else if(id==rb_intern("linear_piece_symmetric")) {
        activation_function = FANN_LINEAR_PIECE_SYMMETRIC;  
    }   else if(id==rb_intern("sin_symmetric")) {
        activation_function = FANN_SIN_SYMMETRIC;   
    }   else if(id==rb_intern("cos_symmetric")) {
        activation_function = FANN_COS_SYMMETRIC;   
    }   else if(id==rb_intern("sin")) {
        activation_function = FANN_SIN; 
    }   else if(id==rb_intern("cos")) {
        activation_function = FANN_COS; 
    }   else {
        rb_raise(rb_eRuntimeError, "Unrecognized activation function: [%s]", rb_id2name(SYM2ID(activation_func)));
    }   
    return activation_function;
}

// Convert FANN enum type for activation function to corresponding ruby symbol:
VALUE activation_function_to_sym(enum fann_activationfunc_enum fn)
{
    VALUE activation_function;
    
    if(fn==FANN_LINEAR) {
        activation_function = ID2SYM(rb_intern("linear"));  
    }   else if(fn==FANN_THRESHOLD) {
        activation_function = ID2SYM(rb_intern("threshold"));   
    }   else if(fn==FANN_THRESHOLD_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("threshold_symmetric")); 
    }   else if(fn==FANN_SIGMOID) {
        activation_function = ID2SYM(rb_intern("sigmoid")); 
    }   else if(fn==FANN_SIGMOID_STEPWISE) {
        activation_function = ID2SYM(rb_intern("sigmoid_stepwise"));    
    }   else if(fn==FANN_SIGMOID_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("sigmoid_symmetric"));   
    }   else if(fn==FANN_SIGMOID_SYMMETRIC_STEPWISE) {
        activation_function = ID2SYM(rb_intern("sigmoid_symmetric_stepwise"));  
    }   else if(fn==FANN_GAUSSIAN) {
        activation_function = ID2SYM(rb_intern("gaussian"));    
    }   else if(fn==FANN_GAUSSIAN_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("gaussian_symmetric"));  
    }   else if(fn==FANN_GAUSSIAN_STEPWISE) {
        activation_function = ID2SYM(rb_intern("gaussian_stepwise"));   
    }   else if(fn==FANN_ELLIOT) {
        activation_function = ID2SYM(rb_intern("elliot"));  
    }   else if(fn==FANN_ELLIOT_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("elliot_symmetric"));    
    }   else if(fn==FANN_LINEAR_PIECE) {
        activation_function = ID2SYM(rb_intern("linear_piece"));    
    }   else if(fn==FANN_LINEAR_PIECE_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("linear_piece_symmetric"));  
    }   else if(fn==FANN_SIN_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("sin_symmetric"));   
    }   else if(fn==FANN_COS_SYMMETRIC) {
        activation_function = ID2SYM(rb_intern("cos_symmetric"));   
    }   else if(fn==FANN_SIN) {
        activation_function = ID2SYM(rb_intern("sin")); 
    }   else if(fn==FANN_COS) {
        activation_function = ID2SYM(rb_intern("cos")); 
    }   else {
        rb_raise(rb_eRuntimeError, "Unrecognized activation function: [%d]", fn);
    }   
    return activation_function;
}


// Unused for now:
static void fann_mark (struct fann* ann){}

// #define DEBUG 1

// Free memory associated with FANN:
static void fann_free (struct fann* ann)
{
  fann_destroy(ann);
    // ("Destroyed FANN network [%d].\n", ann);
}

// Free memory associated with FANN Training data:
static void fann_training_data_free (struct fann_train_data* train_data)
{
  fann_destroy_train(train_data);
    // printf("Destroyed Training data [%d].\n", train_data);
}

// Create wrapper, but don't allocate anything...do that in 
// initialize, so we can construct with args:
static VALUE fann_allocate (VALUE klass)
{
    return Data_Wrap_Struct (klass, fann_mark, fann_free, 0);
}

// Create wrapper, but don't allocate annything...do that in 
// initialize, so we can construct with args:
static VALUE fann_training_data_allocate (VALUE klass)
{
    return Data_Wrap_Struct (klass, fann_mark, fann_training_data_free, 0);
}


// static VALUE invoke_training_callback(VALUE self) 
// {
//   VALUE callback = rb_funcall(self, rb_intern("training_callback"), 0);
//   return callback;
// }

// static int FANN_API internal_callback(struct fann *ann, struct fann_train_data *train, 
//     unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, unsigned int epochs)

static int FANN_API fann_training_callback(struct fann *ann, struct fann_train_data *train,
                           unsigned int max_epochs, unsigned int epochs_between_reports,
                           float desired_error, unsigned int epochs)
{
    VALUE self = (VALUE)fann_get_user_data(ann);
    VALUE args = rb_hash_new();
    
    // Set attributes on hash & push on array:
    VALUE max_epochs_sym = ID2SYM(rb_intern("max_epochs"));
    VALUE epochs_between_reports_sym = ID2SYM(rb_intern("epochs_between_reports"));
    VALUE desired_error_sym = ID2SYM(rb_intern("desired_error"));
    VALUE epochs_sym = ID2SYM(rb_intern("epochs"));
    
    rb_hash_aset(args, max_epochs_sym, INT2NUM(max_epochs));
    rb_hash_aset(args, epochs_between_reports_sym, INT2NUM(epochs_between_reports));
    rb_hash_aset(args, desired_error_sym, rb_float_new(desired_error));
    rb_hash_aset(args, epochs_sym, INT2NUM(epochs));
  
    VALUE callback = rb_funcall(self, rb_intern("training_callback"), 1, args);
  
    if (TYPE(callback)!=T_FIXNUM)
    {
        rb_raise (rb_eRuntimeError, "Callback method must return an integer (-1 to stop training).");
    }

    int status = NUM2INT(callback);   
    if (status==-1)
    {
        printf("Callback method returned -1; training will stop.\n");
    }
    
    return status;
}

/** call-seq: new(hash) -> new ruby-fann neural network object 

    Initialization routine for both standard, shortcut & filename forms of FANN:

    Standard Initialization:
      RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
            
    Shortcut Initialization (e.g., for use in cascade training):
      RubyFann::Shortcut.new(:num_inputs=>5, :num_outputs=>1)    
            
    File Initialization
      RubyFann::Standard.new(:filename=>'xor_float.net') 
      
      
      
*/
static VALUE fann_initialize(VALUE self, VALUE hash)
{
    // Get args:
    VALUE filename = rb_hash_aref(hash, ID2SYM(rb_intern("filename")));
    VALUE num_inputs = rb_hash_aref(hash, ID2SYM(rb_intern("num_inputs")));
    VALUE num_outputs = rb_hash_aref(hash, ID2SYM(rb_intern("num_outputs")));
    VALUE hidden_neurons = rb_hash_aref(hash, ID2SYM(rb_intern("hidden_neurons")));
  // printf("initializing\n\n\n");
    struct fann* ann;
    if (TYPE(filename)==T_STRING) 
    {
        // Initialize with file:
        // train_data = fann_read_train_from_file(StringValuePtr(filename));
        // DATA_PTR(self) = train_data;
        ann = fann_create_from_file(StringValuePtr(filename));
    // printf("Created RubyFann::Standard [%d] from file [%s].\n", ann, StringValuePtr(filename));      
    } 
    else if(rb_obj_is_kind_of(self, m_rb_fann_shortcut_class))
    {
        // Initialize as shortcut, suitable for cascade training:
        //ann = fann_create_shortcut_array(num_layers, layers); 
        Check_Type(num_inputs, T_FIXNUM);
        Check_Type(num_outputs, T_FIXNUM);
        
        ann = fann_create_shortcut(2, NUM2INT(num_inputs), NUM2INT(num_outputs));   
        // printf("Created RubyFann::Shortcut [%d].\n", ann);
    }
    else
    {
        // Initialize as standard:
        Check_Type(num_inputs, T_FIXNUM);
        Check_Type(hidden_neurons, T_ARRAY);
        Check_Type(num_outputs, T_FIXNUM);
        
        // Initialize layers:
        unsigned int num_layers=RARRAY_LEN(hidden_neurons) + 2; 
        unsigned int layers[num_layers];

        // Input:
        layers[0]=NUM2INT(num_inputs); 
        // Output:
        layers[num_layers-1]=NUM2INT(num_outputs);  
        // Hidden:
        int i;
        for (i=1; i<=num_layers-2; i++) {
            layers[i]=NUM2UINT(RARRAY_PTR(hidden_neurons)[i-1]);
        }
        
        ann = fann_create_standard_array(num_layers, layers);   
        // printf("Created RubyFann::Standard [%d].\n", ann);
    }   

    DATA_PTR(self) = ann;
    
    // printf("Checking for callback...");
    
    //int callback = rb_protect(invoke_training_callback, (self), &status);
    // VALUE callback = rb_funcall(DATA_PTR(self), "training_callback", 0);
    if(rb_respond_to(self, rb_intern("training_callback")))
    {
        fann_set_callback(ann, &fann_training_callback);
        fann_set_user_data(ann, self);
        // printf("found(%d).\n", ann->callback);
    }
    else
    {
        // printf("none found.\n");
    }
  
    return (VALUE)ann;  
}

/** call-seq: new(hash) -> new ruby-fann training data object (RubyFann::TrainData)
   
    Initialize in one of the following forms:
  
    # This is a flat file with training data as described in FANN docs.
    RubyFann::TrainData.new(:filename => 'path/to/training_file.train')
  OR
    # Train with inputs (array of arrays) & desired_outputs (array of arrays)
    # inputs & desired outputs should be of same length
    # All sub-arrays on inputs should be of same length
    # All sub-arrays on desired_outputs should be of same length
    # Sub-arrays on inputs & desired_outputs can be different sizes from one another
    RubyFann::TrainData.new(:inputs=>[[0.2, 0.3, 0.4], [0.8, 0.9, 0.7]], :desired_outputs=>[[3.14], [6.33]])    
*/
static VALUE fann_train_data_initialize(VALUE self, VALUE hash)
{
    struct fann_train_data* train_data;
    Check_Type(hash, T_HASH);
  
    VALUE filename = rb_hash_aref(hash, ID2SYM(rb_intern("filename")));
    VALUE inputs = rb_hash_aref(hash, ID2SYM(rb_intern("inputs")));
    VALUE desired_outputs = rb_hash_aref(hash, ID2SYM(rb_intern("desired_outputs")));

    if (TYPE(filename)==T_STRING) 
    {
        train_data = fann_read_train_from_file(StringValuePtr(filename));
        DATA_PTR(self) = train_data;
    } 
    else if (TYPE(inputs)==T_ARRAY) 
    {
        if (TYPE(desired_outputs)!=T_ARRAY)
        {
            rb_raise (rb_eRuntimeError, "[desired_outputs] must be present when [inputs] used.");
        }

        if (RARRAY_LEN(inputs) < 1)
        {
            rb_raise (rb_eRuntimeError, "[inputs/desired_outputs] must contain at least one value.");
        }

        // The data is here, start constructing:
        if(RARRAY_LEN(inputs) != RARRAY_LEN(desired_outputs)) 
        {
            rb_raise (
                rb_eRuntimeError, 
                "Number of inputs must match number of outputs: (%d != %d)", 
                (int)RARRAY_LEN(inputs), 
                (int)RARRAY_LEN(desired_outputs));
        }

        train_data = fann_create_train_from_rb_ary(inputs, desired_outputs);        
        DATA_PTR(self) = train_data;                
    } 
    else 
    {
        rb_raise (rb_eRuntimeError, "Must construct with a filename(string) or inputs/desired_outputs(arrays).  All args passed via hash with symbols as keys.");
    }
    
    return (VALUE)train_data;
}


/** call-seq: save(filename)

    Save to given filename 
*/
static VALUE training_save(VALUE self, VALUE filename)
{
    Check_Type(filename, T_STRING); 
    struct fann_train_data* t;
    Data_Get_Struct (self, struct fann_train_data, t);  
    fann_save_train(t, StringValuePtr(filename));
    return self;    
}

/** Shuffles training data, randomizing the order. 
    This is recommended for incremental training, while it will have no influence during batch training.*/
static VALUE shuffle(VALUE self)
{
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann_train_data, t);  
    fann_shuffle_train_data(t);
    return self;
}

/** Length of training data*/
static VALUE length_train_data(VALUE self)
{
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann_train_data, t);  
    return(UINT2NUM(fann_length_train_data(t)));
    return self;
}

/** call-seq: set_activation_function(activation_func, layer, neuron)

    Set the activation function for neuron number *neuron* in layer number *layer*, 
        counting the input layer as layer 0.  activation_func must be one of the following symbols:
            :linear, :threshold, :threshold_symmetric, :sigmoid, :sigmoid_stepwise, :sigmoid_symmetric, 
            :sigmoid_symmetric_stepwise, :gaussian, :gaussian_symmetric, :gaussian_stepwise, :elliot, 
            :elliot_symmetric, :linear_piece, :linear_piece_symmetric, :sin_symmetric, :cos_symmetric, 
            :sin, :cos*/
static VALUE set_activation_function(VALUE self, VALUE activation_func, VALUE layer, VALUE neuron)
{
    Check_Type(activation_func, T_SYMBOL);
    Check_Type(layer, T_FIXNUM);
    Check_Type(neuron, T_FIXNUM);
    
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_set_activation_function(f, sym_to_activation_function(activation_func), NUM2INT(layer), NUM2INT(neuron));
    return self;
}

/** call-seq: set_activation_function_hidden(activation_func)

    Set the activation function for all of the hidden layers.  activation_func must be one of the following symbols:
            :linear, :threshold, :threshold_symmetric, :sigmoid, :sigmoid_stepwise, :sigmoid_symmetric, 
            :sigmoid_symmetric_stepwise, :gaussian, :gaussian_symmetric, :gaussian_stepwise, :elliot, 
            :elliot_symmetric, :linear_piece, :linear_piece_symmetric, :sin_symmetric, :cos_symmetric, 
            :sin, :cos*/
static VALUE set_activation_function_hidden(VALUE self, VALUE activation_func)
{
    Check_Type(activation_func, T_SYMBOL);
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_set_activation_function_hidden(f, sym_to_activation_function(activation_func));
    return self;
}

/** call-seq: set_activation_function_layer(activation_func, layer)

    Set the activation function for all the neurons in the layer number *layer*, 
        counting the input layer as layer 0.  activation_func must be one of the following symbols:
            :linear, :threshold, :threshold_symmetric, :sigmoid, :sigmoid_stepwise, :sigmoid_symmetric, 
            :sigmoid_symmetric_stepwise, :gaussian, :gaussian_symmetric, :gaussian_stepwise, :elliot, 
            :elliot_symmetric, :linear_piece, :linear_piece_symmetric, :sin_symmetric, :cos_symmetric, 
            :sin, :cos
            
      It is not possible to set activation functions for the neurons in the input layer.
*/          
static VALUE set_activation_function_layer(VALUE self, VALUE activation_func, VALUE layer)
{
    Check_Type(activation_func, T_SYMBOL);
    Check_Type(layer, T_FIXNUM);
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_set_activation_function_layer(f, sym_to_activation_function(activation_func), NUM2INT(layer));
    return self;
}

/** call-seq: get_activation_function(layer) -> return value 
 
    Get the activation function for neuron number *neuron* in layer number *layer*, 
    counting the input layer as layer 0. 

    It is not possible to get activation functions for the neurons in the input layer.    
*/
static VALUE get_activation_function(VALUE self, VALUE layer, VALUE neuron)
{
    Check_Type(layer, T_FIXNUM);
    Check_Type(neuron, T_FIXNUM);
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_type val = fann_get_activation_function(f, NUM2INT(layer), NUM2INT(neuron));
    return activation_function_to_sym(val);
}

/** call-seq: set_activation_function_output(activation_func)

      Set the activation function for the output layer.  activation_func must be one of the following symbols:
            :linear, :threshold, :threshold_symmetric, :sigmoid, :sigmoid_stepwise, :sigmoid_symmetric, 
            :sigmoid_symmetric_stepwise, :gaussian, :gaussian_symmetric, :gaussian_stepwise, :elliot, 
            :elliot_symmetric, :linear_piece, :linear_piece_symmetric, :sin_symmetric, :cos_symmetric, 
            :sin, :cos*/

static VALUE set_activation_function_output(VALUE self, VALUE activation_func)
{
    Check_Type(activation_func, T_SYMBOL);
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_set_activation_function_output(f, sym_to_activation_function(activation_func));
    return self;
}

/** call-seq: get_activation_steepness(layer, neuron) -> return value 
 
    Get the activation steepness for neuron number neuron in layer number layer, counting the input layer as layer 0. 
*/
static VALUE get_activation_steepness(VALUE self, VALUE layer, VALUE neuron)
{
    Check_Type(layer, T_FIXNUM);
    Check_Type(neuron, T_FIXNUM);
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_type val = fann_get_activation_steepness(f, NUM2INT(layer), NUM2INT(neuron));
    return rb_float_new(val);
}

/** call-seq: set_activation_steepness(steepness, layer, neuron)

    Set the activation steepness for neuron number {neuron} in layer number {layer}, 
    counting the input layer as layer 0.*/
static VALUE set_activation_steepness(VALUE self, VALUE steepness, VALUE layer, VALUE neuron)
{
    Check_Type(steepness, T_FLOAT);
    Check_Type(layer, T_FIXNUM);
    Check_Type(neuron, T_FIXNUM);
    
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_set_activation_steepness(f, NUM2DBL(steepness), NUM2INT(layer), NUM2INT(neuron));
    return self;
}

/** call-seq: set_activation_steepness_hidden(arg) -> return value 

    Set the activation steepness in all of the hidden layers.*/
static VALUE set_activation_steepness_hidden(VALUE self, VALUE steepness)
{
    SET_FANN_FLT(steepness, fann_set_activation_steepness_hidden);
}

/** call-seq: set_activation_steepness_layer(steepness, layer)

    Set the activation steepness all of the neurons in layer number *layer*, 
    counting the input layer as layer 0.*/
static VALUE set_activation_steepness_layer(VALUE self, VALUE steepness, VALUE layer)
{
    Check_Type(steepness, T_FLOAT);
    Check_Type(layer, T_FIXNUM);
    
    struct fann* f;
    Data_Get_Struct(self, struct fann, f);
    fann_set_activation_steepness_layer(f, NUM2DBL(steepness), NUM2INT(layer));
    return self;
}

/** call-seq: set_activation_steepness_output(steepness)

    Set the activation steepness in the output layer.*/
static VALUE set_activation_steepness_output(VALUE self, VALUE steepness)
{
    SET_FANN_FLT(steepness, fann_set_activation_steepness_output);
}

/** Returns the bit fail limit used during training.*/
static VALUE get_bit_fail_limit(VALUE self)
{
    RETURN_FANN_DBL(fann_get_bit_fail_limit);
}

/** call-seq: set_bit_fail_limit(bit_fail_limit)

    Sets the bit fail limit used during training.*/
static VALUE set_bit_fail_limit(VALUE self, VALUE bit_fail_limit)
{
    SET_FANN_FLT(bit_fail_limit, fann_set_bit_fail_limit);
}

/** The decay is a small negative valued number which is the factor that the weights 
    should become smaller in each iteration during quickprop training. This is used 
    to make sure that the weights do not become too high during training.*/
static VALUE get_quickprop_decay(VALUE self)
{
    RETURN_FANN_FLT(fann_get_quickprop_decay);
}

/** call-seq: set_quickprop_decay(quickprop_decay)

    Sets the quickprop decay factor*/
static VALUE set_quickprop_decay(VALUE self, VALUE quickprop_decay)
{
    SET_FANN_FLT(quickprop_decay, fann_set_quickprop_decay);
}

/** The mu factor is used to increase and decrease the step-size during quickprop training. 
    The mu factor should always be above 1, since it would otherwise decrease the step-size 
    when it was suppose to increase it. */
static VALUE get_quickprop_mu(VALUE self)
{
    RETURN_FANN_FLT(fann_get_quickprop_mu);
}

/** call-seq: set_quickprop_mu(quickprop_mu)

    Sets the quickprop mu factor.*/
static VALUE set_quickprop_mu(VALUE self, VALUE quickprop_mu)
{
    SET_FANN_FLT(quickprop_mu, fann_set_quickprop_mu);
}

/** The increase factor is a value larger than 1, which is used to 
    increase the step-size during RPROP training.*/
static VALUE get_rprop_increase_factor(VALUE self)
{
    RETURN_FANN_FLT(fann_get_rprop_increase_factor);
}

/** call-seq: set_rprop_increase_factor(rprop_increase_factor)

    The increase factor used during RPROP training. */
static VALUE set_rprop_increase_factor(VALUE self, VALUE rprop_increase_factor)
{
    SET_FANN_FLT(rprop_increase_factor, fann_set_rprop_increase_factor);
}

/** The decrease factor is a value smaller than 1, which is used to decrease the step-size during RPROP training.*/
static VALUE get_rprop_decrease_factor(VALUE self)
{
    RETURN_FANN_FLT(fann_get_rprop_decrease_factor);
}

/** call-seq: set_rprop_decrease_factor(rprop_decrease_factor)

    The decrease factor is a value smaller than 1, which is used to decrease the step-size during RPROP training.*/
static VALUE set_rprop_decrease_factor(VALUE self, VALUE rprop_decrease_factor)
{
    SET_FANN_FLT(rprop_decrease_factor, fann_set_rprop_decrease_factor);
}

/** The minimum step-size is a small positive number determining how small the minimum step-size may be.*/
static VALUE get_rprop_delta_min(VALUE self)
{
    RETURN_FANN_FLT(fann_get_rprop_delta_min);
}

/** call-seq: set_rprop_delta_min(rprop_delta_min)

    The minimum step-size is a small positive number determining how small the minimum step-size may be.*/
static VALUE set_rprop_delta_min(VALUE self, VALUE rprop_delta_min)
{
    SET_FANN_FLT(rprop_delta_min, fann_set_rprop_delta_min);
}

/** The maximum step-size is a positive number determining how large the maximum step-size may be.*/
static VALUE get_rprop_delta_max(VALUE self)
{
    RETURN_FANN_FLT(fann_get_rprop_delta_max);
}

/** call-seq: set_rprop_delta_max(rprop_delta_max)

    The maximum step-size is a positive number determining how large the maximum step-size may be.*/
static VALUE set_rprop_delta_max(VALUE self, VALUE rprop_delta_max)
{
    SET_FANN_FLT(rprop_delta_max, fann_set_rprop_delta_max);
}

/** The initial step-size is a positive number determining the initial step size.*/
static VALUE get_rprop_delta_zero(VALUE self)
{
    RETURN_FANN_FLT(fann_get_rprop_delta_zero);
}

/** call-seq: set_rprop_delta_zero(rprop_delta_zero)

    The initial step-size is a positive number determining the initial step size.*/
static VALUE set_rprop_delta_zero(VALUE self, VALUE rprop_delta_zero)
{
    SET_FANN_FLT(rprop_delta_zero, fann_set_rprop_delta_zero);
}

/** Return array of bias(es)*/
static VALUE get_bias_array(VALUE self)
{
    struct fann* f;
    unsigned int num_layers;
    Data_Get_Struct (self, struct fann, f);
    num_layers = fann_get_num_layers(f);
    unsigned int layers[num_layers];
    fann_get_bias_array(f, layers); 
    
    // Create ruby array & set outputs:
    VALUE arr;
    arr = rb_ary_new();
    int i;
    for (i=0; i<num_layers; i++)
    {
        rb_ary_push(arr, INT2NUM(layers[i]));
    }
    
    return arr;
}

/** The number of fail bits; means the number of output neurons which differ more 
than the bit fail limit (see <fann_get_bit_fail_limit>, <fann_set_bit_fail_limit>). 
The bits are counted in all of the training data, so this number can be higher than
the number of training data.*/
static VALUE get_bit_fail(VALUE self)
{
    RETURN_FANN_INT(fann_get_bit_fail);
}

/** Get the connection rate used when the network was created.*/
static VALUE get_connection_rate(VALUE self)
{
    RETURN_FANN_INT(fann_get_connection_rate);
}

/** call-seq: get_neurons(layer) -> return value 

    Return array<hash> where each array element is a hash
    representing a neuron.  It contains the following keys:
        :activation_function, symbol -- the activation function
        :activation_steepness=float -- The steepness of the activation function
        :sum=float -- The sum of the inputs multiplied with the weights
        :value=float -- The value of the activation fuction applied to the sum
        :connections=array<int> -- indices of connected neurons(inputs)
        
      This could be done more elegantly (e.g., defining more ruby ext classes).
        This method does not directly correlate to anything in FANN, and accesses
        structs that are not guaranteed to not change.              
*/
static VALUE get_neurons(VALUE self, VALUE layer)
{
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    
    struct fann* f;
    unsigned int i;
    Data_Get_Struct (self, struct fann, f);

    VALUE neuron_array = rb_ary_new();
    VALUE activation_function_sym = ID2SYM(rb_intern("activation_function"));
    VALUE activation_steepness_sym = ID2SYM(rb_intern("activation_steepness"));
    VALUE layer_sym = ID2SYM(rb_intern("layer"));
    VALUE sum_sym = ID2SYM(rb_intern("sum"));
    VALUE value_sym = ID2SYM(rb_intern("value"));
    VALUE connections_sym = ID2SYM(rb_intern("connections"));
    unsigned int layer_num = 0;
    
    
    int nuke_bias_neuron = (fann_get_network_type(f)==FANN_NETTYPE_LAYER);
    for(layer_it = f->first_layer; layer_it != f->last_layer; layer_it++)
    {
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
        {
            if (nuke_bias_neuron && (neuron_it==(layer_it->last_neuron)-1)) continue;
            // Create array of connection indicies:
            VALUE connection_array = rb_ary_new();
            for (i = neuron_it->first_con; i < neuron_it->last_con; i++) {              
                rb_ary_push(connection_array, INT2NUM(f->connections[i] - f->first_layer->first_neuron));   
            }

            VALUE neuron = rb_hash_new();
            
            // Set attributes on hash & push on array:
            rb_hash_aset(neuron, activation_function_sym, activation_function_to_sym(neuron_it->activation_function));
            rb_hash_aset(neuron, activation_steepness_sym, rb_float_new(neuron_it->activation_steepness));
            rb_hash_aset(neuron, layer_sym, INT2NUM(layer_num));
            rb_hash_aset(neuron, sum_sym, rb_float_new(neuron_it->sum));
            rb_hash_aset(neuron, value_sym, rb_float_new(neuron_it->value));
            rb_hash_aset(neuron, connections_sym, connection_array);
            
            rb_ary_push(neuron_array, neuron);          
        }
        ++layer_num;
    }

  // switch (fann_get_network_type(ann)) {
  //     case FANN_NETTYPE_LAYER: {
  //         /* Report one bias in each layer except the last */
  //         if (layer_it != ann->last_layer-1)
  //             *bias = 1;
  //         else
  //             *bias = 0;
  //         break;
  //     }
  //     case FANN_NETTYPE_SHORTCUT: {


    return neuron_array;    
}

/** Get list of layers in array format where each element contains number of neurons in that layer*/
static VALUE get_layer_array(VALUE self)
{
    struct fann* f;
    unsigned int num_layers;
    Data_Get_Struct (self, struct fann, f);
    num_layers = fann_get_num_layers(f);
    unsigned int layers[num_layers];
    fann_get_layer_array(f, layers);    
    
    // Create ruby array & set outputs:
    VALUE arr;
    arr = rb_ary_new();
    int i;
    for (i=0; i<num_layers; i++)
    {
        rb_ary_push(arr, INT2NUM(layers[i]));
    }
    
  return arr;
}

/** Reads the mean square error from the network.*/
static VALUE get_MSE(VALUE self)
{
    RETURN_FANN_DBL(fann_get_MSE);
}

/** Resets the mean square error from the network.
    This function also resets the number of bits that fail.*/   
static VALUE reset_MSE(VALUE self)
{
    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_reset_MSE(f);
    return self;    
}

/** Get the type of network.  Returns as ruby symbol (one of :shortcut, :layer)*/
static VALUE get_network_type(VALUE self)
{
    struct fann* f;
    enum fann_nettype_enum net_type;
    VALUE ret_val;
    Data_Get_Struct (self, struct fann, f);

    net_type = fann_get_network_type(f);
    
    if(net_type==FANN_NETTYPE_LAYER) 
    {
        ret_val = ID2SYM(rb_intern("layer")); // (rb_str_new2("FANN_NETTYPE_LAYER"));
    }
    else if(net_type==FANN_NETTYPE_SHORTCUT)
    {
        ret_val = ID2SYM(rb_intern("shortcut")); // (rb_str_new2("FANN_NETTYPE_SHORTCUT"));
    }   
    return ret_val;
}

/** Get the number of input neurons.*/
static VALUE get_num_input(VALUE self)
{
    RETURN_FANN_INT(fann_get_num_input);
}
    
/** Get the number of layers in the network.*/
static VALUE get_num_layers(VALUE self)
{
    RETURN_FANN_INT(fann_get_num_layers);
}

/** Get the number of output neurons.*/
static VALUE get_num_output(VALUE self)
{
    RETURN_FANN_INT(fann_get_num_output);
}

/** Get the total number of connections in the entire network.*/
static VALUE get_total_connections(VALUE self)
{
    RETURN_FANN_INT(fann_get_total_connections);
}

/** Get the total number of neurons in the entire network.*/
static VALUE get_total_neurons(VALUE self)
{
    RETURN_FANN_INT(fann_get_total_neurons);
}

/** call-seq: set_train_error_function(train_error_function)

    Sets the error function used during training.  One of the following symbols:
            :linear, :tanh */
static VALUE set_train_error_function(VALUE self, VALUE train_error_function)
{
    Check_Type(train_error_function, T_SYMBOL);
    
    ID id=SYM2ID(train_error_function);
    enum fann_errorfunc_enum fann_train_error_function;

    if(id==rb_intern("linear")) {
        fann_train_error_function = FANN_ERRORFUNC_LINEAR;  
    }   else if(id==rb_intern("tanh")) {
        fann_train_error_function = FANN_ERRORFUNC_TANH;    
    }   else {
        rb_raise(rb_eRuntimeError, "Unrecognized train error function: [%s]", rb_id2name(SYM2ID(train_error_function)));
    }   

    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_set_train_error_function(f, fann_train_error_function);
    return self;    
}

/** Returns the error function used during training.  One of the following symbols:
            :linear, :tanh*/      
static VALUE get_train_error_function(VALUE self)
{
    struct fann* f;
    enum fann_errorfunc_enum train_error;
    VALUE ret_val;
    Data_Get_Struct (self, struct fann, f);

    train_error = fann_get_train_error_function(f);
    
    if(train_error==FANN_ERRORFUNC_LINEAR) 
    {
        ret_val = ID2SYM(rb_intern("linear")); 
    }
    else if(train_error==FANN_ERRORFUNC_TANH)
    {
        ret_val = ID2SYM(rb_intern("tanh")); 
    }   
    return ret_val;
}

/** call-seq: set_training_algorithm(train_error_function)

    Set the training algorithm.  One of the following symbols:
            :incremental, :batch, :rprop, :quickprop */
static VALUE set_training_algorithm(VALUE self, VALUE train_error_function)
{
    Check_Type(train_error_function, T_SYMBOL);
    
    ID id=SYM2ID(train_error_function);
    enum fann_train_enum fann_train_algorithm;

    if(id==rb_intern("incremental")) {
        fann_train_algorithm = FANN_TRAIN_INCREMENTAL;  
    }   else if(id==rb_intern("batch")) {
        fann_train_algorithm = FANN_TRAIN_BATCH;    
    }   else if(id==rb_intern("rprop")) {
        fann_train_algorithm = FANN_TRAIN_RPROP;    
    }   else if(id==rb_intern("quickprop")) {
        fann_train_algorithm = FANN_TRAIN_QUICKPROP;    
    }   else {
        rb_raise(rb_eRuntimeError, "Unrecognized training algorithm function: [%s]", rb_id2name(SYM2ID(train_error_function)));
    }   

    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_set_training_algorithm(f, fann_train_algorithm);
    return self;    
}

/** Returns the training algorithm.  One of the following symbols:
            :incremental, :batch, :rprop, :quickprop */
static VALUE get_training_algorithm(VALUE self)
{
    struct fann* f;
    enum fann_train_enum fann_train_algorithm;
    VALUE ret_val;
    Data_Get_Struct (self, struct fann, f);

    fann_train_algorithm = fann_get_training_algorithm(f);
    
    if(fann_train_algorithm==FANN_TRAIN_INCREMENTAL) {
        ret_val = ID2SYM(rb_intern("incremental"));
    } else if(fann_train_algorithm==FANN_TRAIN_BATCH) {
        ret_val = ID2SYM(rb_intern("batch")); 
    } else if(fann_train_algorithm==FANN_TRAIN_RPROP) {
        ret_val = ID2SYM(rb_intern("rprop")); 
    } else if(fann_train_algorithm==FANN_TRAIN_QUICKPROP) {
        ret_val = ID2SYM(rb_intern("quickprop")); 
    }   
    return ret_val;
}

/** call-seq: set_train_stop_function(train_stop_function) -> return value 

    Set the training stop function.  One of the following symbols:
            :mse, :bit */
static VALUE set_train_stop_function(VALUE self, VALUE train_stop_function)
{
    Check_Type(train_stop_function, T_SYMBOL);
    ID id=SYM2ID(train_stop_function);
    enum fann_stopfunc_enum fann_train_stop_function;

    if(id==rb_intern("mse")) {
        fann_train_stop_function = FANN_STOPFUNC_MSE;   
    }   else if(id==rb_intern("bit")) {
        fann_train_stop_function = FANN_STOPFUNC_BIT;   
    }   else {
        rb_raise(rb_eRuntimeError, "Unrecognized stop function: [%s]", rb_id2name(SYM2ID(train_stop_function)));
    }   

    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_set_train_stop_function(f, fann_train_stop_function);
    return self;    
}

/** Returns the training stop function.  One of the following symbols:
            :mse, :bit */
static VALUE get_train_stop_function(VALUE self)
{
    struct fann* f;
    enum fann_stopfunc_enum train_stop;
    VALUE ret_val;
    Data_Get_Struct (self, struct fann, f);

    train_stop = fann_get_train_stop_function(f);
    
    if(train_stop==FANN_STOPFUNC_MSE) 
    {
        ret_val = ID2SYM(rb_intern("mse")); // (rb_str_new2("FANN_NETTYPE_LAYER"));
    }
    else if(train_stop==FANN_STOPFUNC_BIT)
    {
        ret_val = ID2SYM(rb_intern("bit")); // (rb_str_new2("FANN_NETTYPE_SHORTCUT"));
    }   
    return ret_val;
}


/** Will print the connections of the ann in a compact matrix, 
        for easy viewing of the internals of the ann. */
static VALUE print_connections(VALUE self)
{
    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_print_connections(f);
    return self;    
}

/** Print current NN parameters to stdout */
static VALUE print_parameters(VALUE self)
{
    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_print_parameters(f);
    return Qnil;
}

/** call-seq: randomize_weights(min_weight, max_weight)

    Give each connection a random weight between *min_weight* and *max_weight* */
static VALUE randomize_weights(VALUE self, VALUE min_weight, VALUE max_weight)
{
    Check_Type(min_weight, T_FLOAT);
    Check_Type(max_weight, T_FLOAT);
    struct fann* f;
    Data_Get_Struct (self, struct fann, f); 
    fann_randomize_weights(f, NUM2DBL(min_weight), NUM2DBL(max_weight));
    return self;    
}

/** call-seq: run(inputs) -> return value 

    Run neural net on array<Float> of inputs with current parameters.  
    Returns array<Float> as output  */
static VALUE run (VALUE self, VALUE inputs)
{
    Check_Type(inputs, T_ARRAY);

  struct fann* f;
    int i;
    fann_type* outputs;
        
    // Convert inputs to type needed for NN:
    unsigned int len = RARRAY_LEN(inputs);
    fann_type fann_inputs[len];
    for (i=0; i<len; i++)
    {
        fann_inputs[i] = NUM2DBL(RARRAY_PTR(inputs)[i]);
    }
    
    
    // Obtain NN & run method:
  Data_Get_Struct (self, struct fann, f);
    outputs = fann_run(f, fann_inputs);

    // Create ruby array & set outputs:
    VALUE arr;
    arr = rb_ary_new();
    unsigned int output_len=fann_get_num_output(f);
    for (i=0; i<output_len; i++)
    {       
        rb_ary_push(arr, rb_float_new(outputs[i]));
    }
    
  return arr;
}

/** call-seq: init_weights(train_data) -> return value 

    Initialize the weights using Widrow + Nguyen's algorithm. */
static VALUE init_weights(VALUE self, VALUE train_data)
{
    
    Check_Type(train_data, T_DATA);
    
    struct fann* f;
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann, f); 
    Data_Get_Struct (train_data, struct fann_train_data, t);    

    fann_init_weights(f, t);    
    return self;    
}

/** call-seq: train(input, expected_output)

    Train with a single input-output pair.
        input - The inputs given to the network
        expected_output - The outputs expected.  */
static VALUE train(VALUE self, VALUE input, VALUE expected_output)
{
    Check_Type(input, T_ARRAY);
    Check_Type(expected_output, T_ARRAY);

    struct fann* f;
    Data_Get_Struct(self, struct fann, f);

    unsigned int num_input = RARRAY_LEN(input);
    unsigned int num_output = RARRAY_LEN(expected_output);

    fann_type data_input[num_input], data_output[num_output];

    int i;

    for (i = 0; i < num_input; i++) {
        data_input[i] = NUM2DBL(RARRAY_PTR(input)[i]);
    }

    for (i = 0; i < num_output; i++) {
        data_output[i] = NUM2DBL(RARRAY_PTR(expected_output)[i]);
    }

    fann_train(f, data_input, data_output);

    return rb_int_new(0);
}

/** call-seq: train_on_data(train_data, max_epochs, epochs_between_reports, desired_error)

    Train with training data created with RubyFann::TrainData.new
        max_epochs - The maximum number of epochs the training should continue
        epochs_between_reports - The number of epochs between printing a status report to stdout.
        desired_error - The desired <get_MSE> or <get_bit_fail>, depending on which stop function
            is chosen by <set_train_stop_function>.  */
static VALUE train_on_data(VALUE self, VALUE train_data, VALUE max_epochs, VALUE epochs_between_reports, VALUE desired_error)
{
    Check_Type(train_data, T_DATA);
    Check_Type(max_epochs, T_FIXNUM);
    Check_Type(epochs_between_reports, T_FIXNUM);
    Check_Type(desired_error, T_FLOAT);
    
    struct fann* f;
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann, f); 
    Data_Get_Struct (train_data, struct fann_train_data, t);    

    unsigned int fann_max_epochs = NUM2INT(max_epochs);
    unsigned int fann_epochs_between_reports = NUM2INT(epochs_between_reports);
    float fann_desired_error = NUM2DBL(desired_error);  
    fann_train_on_data(f, t, fann_max_epochs, fann_epochs_between_reports, fann_desired_error);
    return rb_int_new(0);
}

/** call-seq: train_epoch(train_data) -> return value 

    Train one epoch with a set of training data, created with RubyFann::TrainData.new */
static VALUE train_epoch(VALUE self, VALUE train_data)
{
    Check_Type(train_data, T_DATA);
    struct fann* f;
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann, f); 
    Data_Get_Struct (train_data, struct fann_train_data, t);    
    return rb_float_new(fann_train_epoch(f, t));
}

/** call-seq: test_data(train_data) -> return value 

    Test a set of training data and calculates the MSE for the training data. */
static VALUE test_data(VALUE self, VALUE train_data)
{
    Check_Type(train_data, T_DATA);
    struct fann* f;
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann, f); 
    Data_Get_Struct (train_data, struct fann_train_data, t);    
    return rb_float_new(fann_test_data(f, t));
}

// Returns the position of the decimal point in the ann.
// Only available in fixed-point mode, which we don't need:
// static VALUE get_decimal_point(VALUE self)
// {
//  struct fann* f;
//  Data_Get_Struct (self, struct fann, f);
//  return INT2NUM(fann_get_decimal_point(f));
// }
    
// returns the multiplier that fix point data is multiplied with.

// Only available in fixed-point mode, which we don't need:
// static VALUE get_multiplier(VALUE self)
// {
//  struct fann* f;
//  Data_Get_Struct (self, struct fann, f);
//  return INT2NUM(fann_get_multiplier(f));
// }

/** call-seq: cascadetrain_on_data(train_data, max_neurons, neurons_between_reports, desired_error)

    Perform cascade training with training data created with RubyFann::TrainData.new
        max_epochs - The maximum number of neurons in trained network
        neurons_between_reports - The number of neurons between printing a status report to stdout.
        desired_error - The desired <get_MSE> or <get_bit_fail>, depending on which stop function
        is chosen by <set_train_stop_function>.  */
static VALUE cascadetrain_on_data(VALUE self, VALUE train_data, VALUE max_neurons, VALUE neurons_between_reports, VALUE desired_error)
{
    Check_Type(train_data, T_DATA);
    Check_Type(max_neurons, T_FIXNUM);
    Check_Type(neurons_between_reports, T_FIXNUM);
    Check_Type(desired_error, T_FLOAT);
    
    struct fann* f;
    struct fann_train_data* t; 
    Data_Get_Struct (self, struct fann, f); 
    Data_Get_Struct (train_data, struct fann_train_data, t);    

    unsigned int fann_max_neurons = NUM2INT(max_neurons);
    unsigned int fann_neurons_between_reports = NUM2INT(neurons_between_reports);
    float fann_desired_error = NUM2DBL(desired_error);
    
    fann_cascadetrain_on_data(f, t, fann_max_neurons, fann_neurons_between_reports, fann_desired_error);
    return self;    
}           

/** The cascade output change fraction is a number between 0 and 1 */
static VALUE get_cascade_output_change_fraction(VALUE self)
{
    RETURN_FANN_FLT(fann_get_cascade_output_change_fraction);
}

/** call-seq: set_cascade_output_change_fraction(cascade_output_change_fraction)

    The cascade output change fraction is a number between 0 and 1 */
static VALUE set_cascade_output_change_fraction(VALUE self, VALUE cascade_output_change_fraction)
{
    SET_FANN_FLT(cascade_output_change_fraction, fann_set_cascade_output_change_fraction);
}

/** The number of cascade output stagnation epochs determines the number of epochs training is allowed to 
        continue without changing the MSE by a fraction of <get_cascade_output_change_fraction>. */
static VALUE get_cascade_output_stagnation_epochs(VALUE self)
{
    RETURN_FANN_INT(fann_get_cascade_output_stagnation_epochs);
}

/** call-seq: set_cascade_output_stagnation_epochs(cascade_output_stagnation_epochs)

    The number of cascade output stagnation epochs determines the number of epochs training is allowed to 
        continue without changing the MSE by a fraction of <get_cascade_output_change_fraction>. */
static VALUE set_cascade_output_stagnation_epochs(VALUE self, VALUE cascade_output_stagnation_epochs)
{
    SET_FANN_INT(cascade_output_stagnation_epochs, fann_set_cascade_output_stagnation_epochs);
}

/** The cascade candidate change fraction is a number between 0 and 1 */
static VALUE get_cascade_candidate_change_fraction(VALUE self)
{
    RETURN_FANN_FLT(fann_get_cascade_candidate_change_fraction);
}

/** call-seq: set_cascade_candidate_change_fraction(cascade_candidate_change_fraction)

    The cascade candidate change fraction is a number between 0 and 1 */
static VALUE set_cascade_candidate_change_fraction(VALUE self, VALUE cascade_candidate_change_fraction)
{
    SET_FANN_FLT(cascade_candidate_change_fraction, fann_set_cascade_candidate_change_fraction);
}

/** The number of cascade candidate stagnation epochs determines the number of epochs training is allowed to
        continue without changing the MSE by a fraction of <get_cascade_candidate_change_fraction>. */
static VALUE get_cascade_candidate_stagnation_epochs(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_candidate_stagnation_epochs);
}

/** call-seq: set_cascade_candidate_stagnation_epochs(cascade_candidate_stagnation_epochs)

    The number of cascade candidate stagnation epochs determines the number of epochs training is allowed to
        continue without changing the MSE by a fraction of <get_cascade_candidate_change_fraction>. */
static VALUE set_cascade_candidate_stagnation_epochs(VALUE self, VALUE cascade_candidate_stagnation_epochs)
{
    SET_FANN_UINT(cascade_candidate_stagnation_epochs, fann_set_cascade_candidate_stagnation_epochs);
}                       

/** The weight multiplier is a parameter which is used to multiply the weights from the candidate neuron
        before adding the neuron to the neural network. This parameter is usually between 0 and 1, and is used
        to make the training a bit less aggressive. */
static VALUE get_cascade_weight_multiplier(VALUE self)
{
    RETURN_FANN_DBL(fann_get_cascade_weight_multiplier);
}

/** call-seq: set_cascade_weight_multiplier(cascade_weight_multiplier)

    The weight multiplier is a parameter which is used to multiply the weights from the candidate neuron
        before adding the neuron to the neural network. This parameter is usually between 0 and 1, and is used
        to make the training a bit less aggressive. */
static VALUE set_cascade_weight_multiplier(VALUE self, VALUE cascade_weight_multiplier)
{
    SET_FANN_DBL(cascade_weight_multiplier, fann_set_cascade_weight_multiplier);
}

/** The candidate limit is a limit for how much the candidate neuron may be trained.
        The limit is a limit on the proportion between the MSE and candidate score. */
static VALUE get_cascade_candidate_limit(VALUE self)
{
    RETURN_FANN_DBL(fann_get_cascade_candidate_limit);
}

/** call-seq: set_cascade_candidate_limit(cascade_candidate_limit)

    The candidate limit is a limit for how much the candidate neuron may be trained.
        The limit is a limit on the proportion between the MSE and candidate score. */
static VALUE set_cascade_candidate_limit(VALUE self, VALUE cascade_candidate_limit)
{
    SET_FANN_DBL(cascade_candidate_limit, fann_set_cascade_candidate_limit);
}

/** The maximum out epochs determines the maximum number of epochs the output connections
        may be trained after adding a new candidate neuron. */
static VALUE get_cascade_max_out_epochs(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_max_out_epochs);
}

/** call-seq: set_cascade_max_out_epochs(cascade_max_out_epochs)

    The maximum out epochs determines the maximum number of epochs the output connections
        may be trained after adding a new candidate neuron. */
static VALUE set_cascade_max_out_epochs(VALUE self, VALUE cascade_max_out_epochs)
{
    SET_FANN_UINT(cascade_max_out_epochs, fann_set_cascade_max_out_epochs);
}

/** The maximum candidate epochs determines the maximum number of epochs the input 
        connections to the candidates may be trained before adding a new candidate neuron. */
static VALUE get_cascade_max_cand_epochs(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_max_cand_epochs);
}

/** call-seq: set_cascade_max_cand_epochs(cascade_max_cand_epochs)

    The maximum candidate epochs determines the maximum number of epochs the input 
        connections to the candidates may be trained before adding a new candidate neuron. */
static VALUE set_cascade_max_cand_epochs(VALUE self, VALUE cascade_max_cand_epochs)
{
    SET_FANN_UINT(cascade_max_cand_epochs, fann_set_cascade_max_cand_epochs);
}

/** The number of candidates used during training (calculated by multiplying <get_cascade_activation_functions_count>,
        <get_cascade_activation_steepnesses_count> and <get_cascade_num_candidate_groups>).  */
static VALUE get_cascade_num_candidates(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_num_candidates);
}

/** The number of activation functions in the <get_cascade_activation_functions> array */
static VALUE get_cascade_activation_functions_count(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_activation_functions_count);
}

/** The learning rate is used to determine how aggressive training should be for some of the
        training algorithms (:incremental, :batch, :quickprop).
        Do however note that it is not used in :rprop. 
        The default learning rate is 0.7. */
static VALUE get_learning_rate(VALUE self)
{
    RETURN_FANN_FLT(fann_get_learning_rate);
}

/** call-seq: set_learning_rate(learning_rate) -> return value 

    The learning rate is used to determine how aggressive training should be for some of the
        training algorithms (:incremental, :batch, :quickprop).
        Do however note that it is not used in :rprop. 
        The default learning rate is 0.7. */
static VALUE set_learning_rate(VALUE self, VALUE learning_rate)
{
    SET_FANN_FLT(learning_rate, fann_set_learning_rate);
}

/** Get the learning momentum. */
static VALUE get_learning_momentum(VALUE self)
{
    RETURN_FANN_FLT(fann_get_learning_momentum);
}

/** call-seq: set_learning_momentum(learning_momentum) -> return value 
  
    Set the learning momentum. */
static VALUE set_learning_momentum(VALUE self, VALUE learning_momentum)
{
    SET_FANN_FLT(learning_momentum, fann_set_learning_momentum);
}

/** call-seq: set_cascade_activation_functions(cascade_activation_functions)

    The cascade activation functions is an array of the different activation functions used by
        the candidates.  The default is [:sigmoid, :sigmoid_symmetric, :gaussian, :gaussian_symmetric, :elliot, :elliot_symmetric] */
static VALUE set_cascade_activation_functions(VALUE self, VALUE cascade_activation_functions)
{
    Check_Type(cascade_activation_functions, T_ARRAY);
    struct fann* f;
    Data_Get_Struct (self, struct fann, f); 
    
    unsigned int cnt = RARRAY_LEN(cascade_activation_functions);
    enum fann_activationfunc_enum fann_activation_functions[cnt];
    int i;
    for (i=0; i<cnt; i++)
    {
        fann_activation_functions[i] = sym_to_activation_function(RARRAY_PTR(cascade_activation_functions)[i]);
    }
    
    fann_set_cascade_activation_functions(f, fann_activation_functions, cnt);
    return self;    
}

/** The cascade activation functions is an array of the different activation functions used by
        the candidates.  The default is [:sigmoid, :sigmoid_symmetric, :gaussian, :gaussian_symmetric, :elliot, :elliot_symmetric] */
static VALUE get_cascade_activation_functions(VALUE self)
{
    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    unsigned int cnt = fann_get_cascade_activation_functions_count(f);
    enum fann_activationfunc_enum* fann_functions = fann_get_cascade_activation_functions(f);

    // Create ruby array & set outputs:
    VALUE arr;
    arr = rb_ary_new();
    int i;
    for (i=0; i<cnt; i++)
    {
        rb_ary_push(arr, activation_function_to_sym(fann_functions[i]));
    }

    return arr;
}

/** The number of activation steepnesses in the <get_cascade_activation_functions> array. */
static VALUE get_cascade_activation_steepnesses_count(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_activation_steepnesses_count);
}

/** The number of candidate groups is the number of groups of identical candidates which will be used
        during training. */
static VALUE get_cascade_num_candidate_groups(VALUE self)
{
    RETURN_FANN_UINT(fann_get_cascade_num_candidate_groups);
}

/** call-seq: set_cascade_num_candidate_groups(cascade_num_candidate_groups)

    The number of candidate groups is the number of groups of identical candidates which will be used
        during training. */
static VALUE set_cascade_num_candidate_groups(VALUE self, VALUE cascade_num_candidate_groups)
{
    SET_FANN_UINT(cascade_num_candidate_groups, fann_set_cascade_num_candidate_groups);
    return 0;
}

/** The cascade activation steepnesses array is an array of the different activation functions used by
        the candidates. */
static VALUE set_cascade_activation_steepnesses(VALUE self, VALUE cascade_activation_steepnesses)
{
    Check_Type(cascade_activation_steepnesses, T_ARRAY);
    struct fann* f;
    Data_Get_Struct (self, struct fann, f); 
    
    unsigned int cnt = RARRAY_LEN(cascade_activation_steepnesses);
    fann_type fann_activation_steepnesses[cnt];
    int i;
    for (i=0; i<cnt; i++)
    {
        fann_activation_steepnesses[i] = NUM2DBL(RARRAY_PTR(cascade_activation_steepnesses)[i]);
    }
    
    fann_set_cascade_activation_steepnesses(f, fann_activation_steepnesses, cnt);
    return self;
}

/** The cascade activation steepnesses array is an array of the different activation functions used by
        the candidates. */
static VALUE get_cascade_activation_steepnesses(VALUE self)
{
    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    fann_type* fann_steepnesses = fann_get_cascade_activation_steepnesses(f);
    unsigned int cnt = fann_get_cascade_activation_steepnesses_count(f);

    // Create ruby array & set outputs:
    VALUE arr;
    arr = rb_ary_new();
    int i;
    for (i=0; i<cnt; i++)
    {
        rb_ary_push(arr, rb_float_new(fann_steepnesses[i]));
    }

    return arr;
}

/** call-seq: save(filename) -> return status

    Save the entire network to configuration file with given name */
static VALUE nn_save(VALUE self, VALUE filename)
{
    struct fann* f;
    Data_Get_Struct (self, struct fann, f);
    int status = fann_save(f, StringValuePtr(filename));
    return INT2NUM(status);
}

/** Initializes class under RubyFann module/namespace. */
void Init_ruby_fann ()
{
    // RubyFann module/namespace:
    m_rb_fann_module = rb_define_module ("RubyFann");

    // Standard NN class:
    m_rb_fann_standard_class = rb_define_class_under (m_rb_fann_module, "Standard", rb_cObject);
    rb_define_alloc_func (m_rb_fann_standard_class, fann_allocate);
    rb_define_method(m_rb_fann_standard_class, "initialize", fann_initialize, 1);
    rb_define_method(m_rb_fann_standard_class, "init_weights", init_weights, 1);
    rb_define_method(m_rb_fann_standard_class, "set_activation_function", set_activation_function, 3);  
    rb_define_method(m_rb_fann_standard_class, "set_activation_function_hidden", set_activation_function_hidden, 1);    
    rb_define_method(m_rb_fann_standard_class, "set_activation_function_layer", set_activation_function_layer, 2);  
    rb_define_method(m_rb_fann_standard_class, "get_activation_function", get_activation_function, 2);  
    rb_define_method(m_rb_fann_standard_class, "set_activation_function_output", set_activation_function_output, 1);    
    rb_define_method(m_rb_fann_standard_class, "get_activation_steepness", get_activation_steepness, 2);
    rb_define_method(m_rb_fann_standard_class, "set_activation_steepness", set_activation_steepness, 3);
    rb_define_method(m_rb_fann_standard_class, "set_activation_steepness_hidden", set_activation_steepness_hidden, 1);
    rb_define_method(m_rb_fann_standard_class, "set_activation_steepness_layer", set_activation_steepness_layer, 2);
    rb_define_method(m_rb_fann_standard_class, "set_activation_steepness_output", set_activation_steepness_output, 1);
    rb_define_method(m_rb_fann_standard_class, "get_train_error_function", get_train_error_function, 0);
    rb_define_method(m_rb_fann_standard_class, "set_train_error_function", set_train_error_function, 1);
    rb_define_method(m_rb_fann_standard_class, "get_train_stop_function", get_train_stop_function, 0);
    rb_define_method(m_rb_fann_standard_class, "set_train_stop_function", set_train_stop_function, 1);
    rb_define_method(m_rb_fann_standard_class, "get_bit_fail_limit", get_bit_fail_limit, 0);
    rb_define_method(m_rb_fann_standard_class, "set_bit_fail_limit", set_bit_fail_limit, 1);
    rb_define_method(m_rb_fann_standard_class, "get_quickprop_decay", get_quickprop_decay, 0);
    rb_define_method(m_rb_fann_standard_class, "set_quickprop_decay", set_quickprop_decay, 1);
    rb_define_method(m_rb_fann_standard_class, "get_quickprop_mu", get_quickprop_mu, 0);
    rb_define_method(m_rb_fann_standard_class, "set_quickprop_mu", set_quickprop_mu, 1);
    rb_define_method(m_rb_fann_standard_class, "get_rprop_increase_factor", get_rprop_increase_factor, 0);
    rb_define_method(m_rb_fann_standard_class, "set_rprop_increase_factor", set_rprop_increase_factor, 1);
    rb_define_method(m_rb_fann_standard_class, "get_rprop_decrease_factor", get_rprop_decrease_factor, 0);
    rb_define_method(m_rb_fann_standard_class, "set_rprop_decrease_factor", set_rprop_decrease_factor, 1);
    rb_define_method(m_rb_fann_standard_class, "get_rprop_delta_max", get_rprop_delta_max, 0);
    rb_define_method(m_rb_fann_standard_class, "set_rprop_delta_max", set_rprop_delta_max, 1);
    rb_define_method(m_rb_fann_standard_class, "get_rprop_delta_min", get_rprop_delta_min, 0);
    rb_define_method(m_rb_fann_standard_class, "set_rprop_delta_min", set_rprop_delta_min, 1);
    rb_define_method(m_rb_fann_standard_class, "get_rprop_delta_zero", get_rprop_delta_zero, 0);
    rb_define_method(m_rb_fann_standard_class, "set_rprop_delta_zero", set_rprop_delta_zero, 1);
    rb_define_method(m_rb_fann_standard_class, "get_bias_array", get_bias_array, 0);
    rb_define_method(m_rb_fann_standard_class, "get_connection_rate", get_connection_rate, 0);
    rb_define_method(m_rb_fann_standard_class, "get_layer_array", get_layer_array, 0);
    rb_define_method(m_rb_fann_standard_class, "get_network_type", get_network_type, 0);
    rb_define_method(m_rb_fann_standard_class, "get_neurons", get_neurons, 0);  
    rb_define_method(m_rb_fann_standard_class, "get_num_input", get_num_input, 0);
    rb_define_method(m_rb_fann_standard_class, "get_num_layers", get_num_layers, 0);
    rb_define_method(m_rb_fann_standard_class, "get_num_output", get_num_output, 0);    
    rb_define_method(m_rb_fann_standard_class, "get_total_connections", get_total_connections, 0);
    rb_define_method(m_rb_fann_standard_class, "get_total_neurons", get_total_neurons, 0);
    // rb_define_method(m_rb_fann_standard_class, "get_train_error_function", get_train_error_function, 0);
    // rb_define_method(m_rb_fann_standard_class, "set_train_error_function", set_train_error_function, 1);    
    rb_define_method(m_rb_fann_standard_class, "print_connections", print_connections, 0);
    rb_define_method(m_rb_fann_standard_class, "print_parameters", print_parameters, 0);
    rb_define_method(m_rb_fann_standard_class, "randomize_weights", randomize_weights, 2);
    rb_define_method(m_rb_fann_standard_class, "run", run, 1);
    rb_define_method(m_rb_fann_standard_class, "train", train, 2);
    rb_define_method(m_rb_fann_standard_class, "train_on_data", train_on_data, 4);
    rb_define_method(m_rb_fann_standard_class, "train_epoch", train_epoch, 1);
    rb_define_method(m_rb_fann_standard_class, "test_data", test_data, 1);  
    rb_define_method(m_rb_fann_standard_class, "get_MSE", get_MSE, 0);
    rb_define_method(m_rb_fann_standard_class, "get_bit_fail", get_bit_fail, 0);
    rb_define_method(m_rb_fann_standard_class, "reset_MSE", reset_MSE, 0);
    rb_define_method(m_rb_fann_standard_class, "get_learning_rate", get_learning_rate, 0);
    rb_define_method(m_rb_fann_standard_class, "set_learning_rate", set_learning_rate, 1);
    rb_define_method(m_rb_fann_standard_class, "get_learning_momentum", get_learning_momentum, 0);
    rb_define_method(m_rb_fann_standard_class, "set_learning_momentum", set_learning_momentum, 1);
    rb_define_method(m_rb_fann_standard_class, "get_training_algorithm", get_training_algorithm, 0);
    rb_define_method(m_rb_fann_standard_class, "set_training_algorithm", set_training_algorithm, 1);
    
    
    // Cascade functions:
    rb_define_method(m_rb_fann_standard_class, "cascadetrain_on_data", cascadetrain_on_data, 4);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_output_change_fraction", get_cascade_output_change_fraction, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_output_change_fraction", set_cascade_output_change_fraction, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_output_stagnation_epochs", get_cascade_output_stagnation_epochs, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_output_stagnation_epochs", set_cascade_output_stagnation_epochs, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_candidate_change_fraction", get_cascade_candidate_change_fraction, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_candidate_change_fraction", set_cascade_candidate_change_fraction, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_candidate_stagnation_epochs", get_cascade_candidate_stagnation_epochs, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_candidate_stagnation_epochs", set_cascade_candidate_stagnation_epochs, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_weight_multiplier", get_cascade_weight_multiplier, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_weight_multiplier", set_cascade_weight_multiplier, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_candidate_limit", get_cascade_candidate_limit, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_candidate_limit", set_cascade_candidate_limit, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_max_out_epochs", get_cascade_max_out_epochs, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_max_out_epochs", set_cascade_max_out_epochs, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_max_cand_epochs", get_cascade_max_cand_epochs, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_max_cand_epochs", set_cascade_max_cand_epochs, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_num_candidates", get_cascade_num_candidates, 0);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_activation_functions_count", get_cascade_activation_functions_count, 0);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_activation_functions", get_cascade_activation_functions, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_activation_functions", set_cascade_activation_functions, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_activation_steepnesses_count", get_cascade_activation_steepnesses_count, 0);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_activation_steepnesses", get_cascade_activation_steepnesses, 0);
    rb_define_method(m_rb_fann_standard_class, "set_cascade_activation_steepnesses", set_cascade_activation_steepnesses, 1);
    rb_define_method(m_rb_fann_standard_class, "get_cascade_num_candidate_groups", get_cascade_num_candidate_groups, 0);    
    rb_define_method(m_rb_fann_standard_class, "set_cascade_num_candidate_groups", set_cascade_num_candidate_groups, 1);    
    rb_define_method(m_rb_fann_standard_class, "save", nn_save, 1);

    
    // Uncomment for fixed-point mode (also recompile fann).  Probably not going to be needed:
    //rb_define_method(clazz, "get_decimal_point", get_decimal_point, 0);   
    //rb_define_method(clazz, "get_multiplier", get_multiplier, 0); 
    
    // Shortcut NN class (duplicated from above so that rdoc generation tools can find the methods:):
    m_rb_fann_shortcut_class = rb_define_class_under (m_rb_fann_module, "Shortcut", rb_cObject);    
    rb_define_alloc_func (m_rb_fann_shortcut_class, fann_allocate);
    rb_define_method(m_rb_fann_shortcut_class, "initialize", fann_initialize, 1);
    rb_define_method(m_rb_fann_shortcut_class, "init_weights", init_weights, 1);
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_function", set_activation_function, 3);  
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_function_hidden", set_activation_function_hidden, 1);    
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_function_layer", set_activation_function_layer, 2);  
    rb_define_method(m_rb_fann_shortcut_class, "get_activation_function", get_activation_function, 2);  
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_function_output", set_activation_function_output, 1);    
    rb_define_method(m_rb_fann_shortcut_class, "get_activation_steepness", get_activation_steepness, 2);
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_steepness", set_activation_steepness, 3);
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_steepness_hidden", set_activation_steepness_hidden, 1);
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_steepness_layer", set_activation_steepness_layer, 2);
    rb_define_method(m_rb_fann_shortcut_class, "set_activation_steepness_output", set_activation_steepness_output, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_train_error_function", get_train_error_function, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_train_error_function", set_train_error_function, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_train_stop_function", get_train_stop_function, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_train_stop_function", set_train_stop_function, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_bit_fail_limit", get_bit_fail_limit, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_bit_fail_limit", set_bit_fail_limit, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_quickprop_decay", get_quickprop_decay, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_quickprop_decay", set_quickprop_decay, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_quickprop_mu", get_quickprop_mu, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_quickprop_mu", set_quickprop_mu, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_rprop_increase_factor", get_rprop_increase_factor, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_rprop_increase_factor", set_rprop_increase_factor, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_rprop_decrease_factor", get_rprop_decrease_factor, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_rprop_decrease_factor", set_rprop_decrease_factor, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_rprop_delta_max", get_rprop_delta_max, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_rprop_delta_max", set_rprop_delta_max, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_rprop_delta_min", get_rprop_delta_min, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_rprop_delta_min", set_rprop_delta_min, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_rprop_delta_zero", get_rprop_delta_zero, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_rprop_delta_zero", set_rprop_delta_zero, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_bias_array", get_bias_array, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_connection_rate", get_connection_rate, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_layer_array", get_layer_array, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_network_type", get_network_type, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_neurons", get_neurons, 0);  
    rb_define_method(m_rb_fann_shortcut_class, "get_num_input", get_num_input, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_num_layers", get_num_layers, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_num_output", get_num_output, 0);    
    rb_define_method(m_rb_fann_shortcut_class, "get_total_connections", get_total_connections, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_total_neurons", get_total_neurons, 0);
    // rb_define_method(m_rb_fann_shortcut_class, "get_train_error_function", get_train_error_function, 0);
    // rb_define_method(m_rb_fann_shortcut_class, "set_train_error_function", set_train_error_function, 1);    
    rb_define_method(m_rb_fann_shortcut_class, "print_connections", print_connections, 0);
    rb_define_method(m_rb_fann_shortcut_class, "print_parameters", print_parameters, 0);
    rb_define_method(m_rb_fann_shortcut_class, "randomize_weights", randomize_weights, 2);
    rb_define_method(m_rb_fann_shortcut_class, "run", run, 1);
    rb_define_method(m_rb_fann_shortcut_class, "train", train, 2);
    rb_define_method(m_rb_fann_shortcut_class, "train_on_data", train_on_data, 4);
    rb_define_method(m_rb_fann_shortcut_class, "train_epoch", train_epoch, 1);
    rb_define_method(m_rb_fann_shortcut_class, "test_data", test_data, 1);  
    rb_define_method(m_rb_fann_shortcut_class, "get_MSE", get_MSE, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_bit_fail", get_bit_fail, 0);
    rb_define_method(m_rb_fann_shortcut_class, "reset_MSE", reset_MSE, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_learning_rate", get_learning_rate, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_learning_rate", set_learning_rate, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_learning_momentum", get_learning_momentum, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_learning_momentum", set_learning_momentum, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_training_algorithm", get_training_algorithm, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_training_algorithm", set_training_algorithm, 1);
    
    // Cascade functions:
    rb_define_method(m_rb_fann_shortcut_class, "cascadetrain_on_data", cascadetrain_on_data, 4);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_output_change_fraction", get_cascade_output_change_fraction, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_output_change_fraction", set_cascade_output_change_fraction, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_output_stagnation_epochs", get_cascade_output_stagnation_epochs, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_output_stagnation_epochs", set_cascade_output_stagnation_epochs, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_candidate_change_fraction", get_cascade_candidate_change_fraction, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_candidate_change_fraction", set_cascade_candidate_change_fraction, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_candidate_stagnation_epochs", get_cascade_candidate_stagnation_epochs, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_candidate_stagnation_epochs", set_cascade_candidate_stagnation_epochs, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_weight_multiplier", get_cascade_weight_multiplier, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_weight_multiplier", set_cascade_weight_multiplier, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_candidate_limit", get_cascade_candidate_limit, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_candidate_limit", set_cascade_candidate_limit, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_max_out_epochs", get_cascade_max_out_epochs, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_max_out_epochs", set_cascade_max_out_epochs, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_max_cand_epochs", get_cascade_max_cand_epochs, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_max_cand_epochs", set_cascade_max_cand_epochs, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_num_candidates", get_cascade_num_candidates, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_activation_functions_count", get_cascade_activation_functions_count, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_activation_functions", get_cascade_activation_functions, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_activation_functions", set_cascade_activation_functions, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_activation_steepnesses_count", get_cascade_activation_steepnesses_count, 0);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_activation_steepnesses", get_cascade_activation_steepnesses, 0);
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_activation_steepnesses", set_cascade_activation_steepnesses, 1);
    rb_define_method(m_rb_fann_shortcut_class, "get_cascade_num_candidate_groups", get_cascade_num_candidate_groups, 0);    
    rb_define_method(m_rb_fann_shortcut_class, "set_cascade_num_candidate_groups", set_cascade_num_candidate_groups, 1);    
    rb_define_method(m_rb_fann_shortcut_class, "save", nn_save, 1);
    

    // TrainData NN class:
    m_rb_fann_train_data_class = rb_define_class_under (m_rb_fann_module, "TrainData", rb_cObject); 
    rb_define_alloc_func (m_rb_fann_train_data_class, fann_training_data_allocate);     
    rb_define_method(m_rb_fann_train_data_class, "initialize", fann_train_data_initialize, 1);
    rb_define_method(m_rb_fann_train_data_class, "length", length_train_data, 0);
    rb_define_method(m_rb_fann_train_data_class, "shuffle", shuffle, 0);    
    rb_define_method(m_rb_fann_train_data_class, "save", training_save, 1);
    
    // printf("Initialized Ruby Bindings for FANN.\n");
}

