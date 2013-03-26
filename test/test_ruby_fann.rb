require 'test/unit'
require 'rubygems'
require 'ruby_fann/neural_network'

class MyShortcut < RubyFann::Shortcut
  def initialize
    super(:num_inputs=>5, :num_outputs=>1)    
  end
end

class MyFann < RubyFann::Standard
  attr_accessor :callback_invoked
  # def initialize
  #   super(:num_inputs=>5, :num_outputs=>1)    
  # end
  def training_callback(args)
    puts "ARGS: #{args.inspect}"
    @callback_invoked=true
    0
  end
end




class RubyFannTest < Test::Unit::TestCase
  def test_create_standard
    fann = RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)    
    assert(fann)
  end
  
  def test_create_shortcut
    fann = RubyFann::Shortcut.new(:num_inputs=>5, :num_outputs=>1)    
    
    assert(fann)
    
    assert_equal(2, fann.get_num_layers)
    assert_equal(5, fann.get_num_input)
    assert_equal(1, fann.get_num_output)    

    fann2 = RubyFann::Shortcut.new(:num_inputs=>1, :num_outputs=>2)    
    assert(fann2)
    assert_equal(2, fann2.get_num_layers)
    assert_equal(1, fann2.get_num_input)
    assert_equal(2, fann2.get_num_output)    
    assert_equal(:shortcut, fann.get_network_type)
        
    sc = MyShortcut.new
  end

  def test_raises
    fann = nil
    assert_raises(TypeError) { fann = RubyFann::Standard.new(:num_inputs=>'foo', :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)}
    assert_nil(fann)    
    assert_raises(ArgumentError)  { fann = RubyFann::Standard.new(4, 1, [3, 4], 1) }        
    assert_nil(fann)    
    assert_nothing_raised { fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1) }    
    assert(fann)
  end
  
  def test_run
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    outputs = fann.run([3.0, 2.0, 3.0, 4.0])
    assert_equal(1, outputs.length)
  end
  
  def test_print_parameters
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    fann.print_parameters
  end
  
  def test_accessors
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3], :num_outputs=>1)
    assert_equal(5, fann.get_num_layers)
    assert_equal(4, fann.get_num_input)
    assert_equal(1, fann.get_num_output)
    puts "TN: #{fann.get_total_neurons}"
    puts "TC: #{fann.get_total_connections}"
    fann.print_parameters
    assert_equal(19, fann.get_total_neurons)
  end
  
  def test_randomize_weights
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    fann.randomize_weights(-1.0, 1.0)
  end
  
  def test_init_weights
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    training = RubyFann::TrainData.new(:filename=>'test/test.train')    
    fann.init_weights(training)
  end
  
  
  def test_print_connections
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    fann.print_connections
  end
  
  def test_connection_rate
    fann = RubyFann::Standard.new(:num_inputs=>4, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    cr = fann.get_connection_rate
    assert_equal(1, cr)    
  end
  
  def test_layer_array
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    layers = fann.get_layer_array
    assert_equal([5, 2, 8, 4, 3, 4, 1], layers)
  end
  
  def test_bias_array
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 4, 3, 4], :num_outputs=>1) 
    bias = fann.get_bias_array
    assert_equal([1, 1, 1, 1, 1, 0], bias)
  end
  
  
  def test_get_network_type
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    assert_equal(:layer, fann.get_network_type)
    puts "fann.get_network_type: #{fann.get_network_type}"
  end

  def test_train_file
    training = RubyFann::TrainData.new(:filename=>'test/test.train')
    training.save('verify.train')
    assert(File.exist?('verify.train'))  	
    test_train_txt = File.read('test/test.train')
    verify_train_txt = File.read('verify.train')
    assert_equal(verify_train_txt, test_train_txt)    
  end 
  
  def test_train_creation_data
    training = RubyFann::TrainData.new(:inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]], :desired_outputs=>[[0.7], [0.8]])
    training.save('verify.train')
    verify_lines = File.readlines('verify.train')
    assert_equal(5, verify_lines.length)
    assert_match(/2 3 1/, verify_lines[0])
    assert_match(/0.30* 0.40* 0.50*/, verify_lines[1])
    assert_match(/0.7/, verify_lines[2])
    assert_match(/0.10* 0.20* 0.30*/, verify_lines[3])
    assert_match(/0.8/, verify_lines[4])
  end
  
  def test_train_on_data
    train = RubyFann::TrainData.new(:inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]], :desired_outputs=>[[0.7], [0.8]])
    fann = RubyFann::Standard.new(:num_inputs=>3, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)
    fann.train_on_data(train, 1000, 10, 0.1)
    outputs = fann.run([3.0, 2.0, 3.0])    
    puts "OUTPUT FROM RUN WAS #{outputs.inspect}"
  end
  
  def test_train_callback
    puts "train callback"
    train = RubyFann::TrainData.new(:inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]], :desired_outputs=>[[0.7], [0.8]])
    fann = MyFann.new(:num_inputs=>3, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)
    
    assert(!fann.callback_invoked)
    fann.train_on_data(train, 1000, 1, 0.01)    
    assert(fann.callback_invoked)
  end
  
  def test_train_bug
    require 'rubygems'
    require 'ruby_fann/neural_network'
    training_data = RubyFann::TrainData.new(
      :inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]],
      :desired_outputs=>[[0.7], [0.8]])
    
    fann = RubyFann::Standard.new(
      :num_inputs=>3,
      :hidden_neurons=>[2, 8, 4, 3, 4],
      :num_outputs=>1)
      
    fann.train_on_data(training_data, 1000, 1, 0.1)
  end
  
  def test_activation_function
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann.set_activation_function(:linear, 1, 2)
    assert_raises(TypeError) { fann.set_activation_function('linear', 1, 2) }
  end  
  
  def test_activation_function_hidden
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)
    fann.set_activation_function_hidden(:linear)
    assert_raises(RuntimeError) { fann.set_activation_function_hidden(:fake) }
    fann.set_activation_function_hidden(:elliot)
  end
  
  def test_activation_function_layer
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)
    fann.set_activation_function_layer(:linear, 1)
    assert_raises(RuntimeError) { fann.set_activation_function_layer(:fake, 0) }
    assert_equal(:linear, fann.get_activation_function(1, 0))
  end

  def test_activation_function_output
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)  
    fann.set_activation_function_output(:linear)                                                      
    assert_raises(RuntimeError) { fann.set_activation_function_output(:fake) }                        
    fann.set_activation_function_output(:elliot)                                                      
  end                                                                                                 
                                                                                                      
  def test_activation_steepness                                                                       
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)  
    fann.set_activation_steepness(0.2, 2, 3)                                                          
    assert_equal(0.2, fann.get_activation_steepness(2, 3))                                            
  end	                                                                                                
                                                                                                      
  def test_activation_steepness_hidden                                                                
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)  
    fann.set_activation_steepness_hidden(0.345)                                                       
    1.upto(5) {|i|assert_equal(0.345, fann.get_activation_steepness(i, 1)) }                          
  end                                                                                                 
                                                                                                      
  def test_activation_steepness_layer                                                                 
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)  
    fann.set_activation_steepness_layer(0.7000, 3)                                                    
    assert_equal(0.7000, fann.get_activation_steepness(3, 1))                                         
  end                                                                                                 
                                                                                                      
  def test_activation_steepness_output                                                                
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann.set_activation_steepness_output(0.888)                                                       
    assert_equal(0.888, fann.get_activation_steepness(6, 1))                                          
  end                                                                                                 
                                                                                                      
  def test_training_function                                                                          
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann.set_train_error_function(:tanh)                                                              
    assert_equal(:tanh, fann.get_train_error_function())                                              
                                                                                                      
    fann2 = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann2.set_train_error_function(:linear)                                                           
    assert_equal(:linear, fann2.get_train_error_function())                                           
    assert_raises(RuntimeError) { fann2.set_train_error_function(:fake) }                             
  end                                                                                                 
                                                                                                      
  def test_training_stop                                                                              
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann.set_train_stop_function(:mse)                                                                
    assert_equal(:mse, fann.get_train_stop_function())                                                
                                                                                                      
    fann2 = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann2.set_train_stop_function(:bit)                                                               
    assert_equal(:bit, fann2.get_train_stop_function())                                               
                                                                                                      
    assert_raises(RuntimeError) { fann2.set_train_stop_function(:fake) }                              
  end                                                                                                 
                                                                                                      
  def test_save                                                                                       
    fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1) 
    fann.save('foo.net')
    assert(File.exist?('foo.net'))
    File.delete('foo.net')
  end
  
  def test_training_algorithm
    verify_fann_attribute(:training_algorithm, :rprop)
  end
  
  def test_bit_fail_limit
    verify_fann_attribute(:bit_fail_limit, 0.743)
  end

  def test_quickprop_decay
    verify_fann_attribute(:quickprop_decay, 0.211)
  end

  def test_quickprop_mu
    verify_fann_attribute(:quickprop_mu, 0.912)
  end

  def test_rprop_increase_factor
    verify_fann_attribute(:rprop_increase_factor, 0.743)
  end

  def test_rprop_decrease_factor
    verify_fann_attribute(:rprop_decrease_factor, 0.190)
  end

  def test_rprop_delta_min
    verify_fann_attribute(:rprop_delta_min, 0.277)
  end
  
  def test_rprop_delta_max
    verify_fann_attribute(:rprop_delta_max, 0.157)
  end
  
  def test_rprop_delta_zero
    verify_fann_attribute(:rprop_delta_zero, 0.571)
  end

  def test_learning_momentum
    verify_fann_attribute(:learning_momentum, 0.231)
  end

  def test_learning_rate
    verify_fann_attribute(:learning_rate, 0.012)
  end
  
  def test_cascade_train
    train = RubyFann::TrainData.new(:inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]], :desired_outputs=>[[0.7], [0.8]])
    fann = RubyFann::Shortcut.new(:num_inputs=>5, :num_outputs=>1) # RubyFann::Shortcut.new(7, 5, [2, 8, 4, 3, 4], 1)    
    fann.cascadetrain_on_data(train, 1000, 10, 0.1)    
  end
  
  def test_get_neurons
    train = RubyFann::TrainData.new(:inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]], :desired_outputs=>[[0.7, 0.3], [0.8, 0.4]])
    fann = RubyFann::Shortcut.new(:num_inputs=>5, :num_outputs=>2) # RubyFann::Shortcut.new(7, 5, [2, 8, 4, 3, 4], 1)    
    fann.cascadetrain_on_data(train, 30, 10, 0.0000000000001)    
    neurons = fann.get_neurons()
    assert_equal(38, neurons.length)
    neurons.each_with_index {|n, i| puts "NEURON[#{i}]: #{n.inspect}" }
  end

  def test_cascade_output_change_fraction
    verify_fann_attribute(:cascade_output_change_fraction, 0.222)
  end

  def test_cascade_output_stagnation_epochs
    verify_fann_attribute(:cascade_output_stagnation_epochs, 4)
  end

  def test_cascade_candidate_change_fraction
    verify_fann_attribute(:cascade_candidate_change_fraction, 0.987)
  end

  def test_cascade_candidate_stagnation_epochs
    verify_fann_attribute(:cascade_candidate_stagnation_epochs, 5)
  end

  def test_cascade_weight_multiplier
    verify_fann_attribute(:cascade_weight_multiplier, 0.754)
  end

  def test_cascade_candidate_limit
    verify_fann_attribute(:cascade_candidate_limit, 0.222)
  end
  
  def test_cascade_cascade_max_out_epochs
    verify_fann_attribute(:cascade_max_out_epochs, 77)
  end

  def test_cascade_max_cand_epochs
    verify_fann_attribute(:cascade_max_cand_epochs, 66)
  end
  
  def test_cascade_num_candidate_groups
    verify_fann_attribute(:cascade_num_candidate_groups, 6)
  end
    
  def test_cascade_num_candidates
    fann = RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)     # RubyFann::Standard.new(7, 5, [2, 8, 4, 3, 4], 1)    
    x = fann.get_cascade_activation_functions_count
    y = fann.get_cascade_activation_steepnesses_count
    z = fann.get_cascade_num_candidate_groups
    assert_equal((x * y * z), fann.get_cascade_num_candidates)
  end
  
  def test_cascade_activation_functions
    fann = RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)     # RubyFann::Standard.new(7, 5, [2, 8, 4, 3, 4], 1)    
    fann.set_cascade_activation_functions([:threshold, :sigmoid])
    assert_equal(2, fann.get_cascade_activation_functions_count)
    assert_equal([:threshold, :sigmoid], fann.get_cascade_activation_functions)
  end

  def test_cascade_activation_steepness
    fann = RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)     # RubyFann::Standard.new(7, 5, [2, 8, 4, 3, 4], 1)    
    fann.set_cascade_activation_steepnesses([0.1, 0.3, 0.7, 0.5])
    assert_equal(4, fann.get_cascade_activation_steepnesses_count)
    assert_equal([0.1, 0.3, 0.7, 0.5], fann.get_cascade_activation_steepnesses)
  end
    
  def test_train_exception
    # Mismatched inputs & outputs
    assert_raises(RuntimeError) { 
      RubyFann::TrainData.new(
        { 
          :inputs=>[[0.2, 0.3, 0.4], [0.8, 0.9, 0.7]], 
          :desired_outputs=>[[3.14]]
        }
      ) 
    } 
    
    # Wrong arg type:
    assert_raises(TypeError) { RubyFann::TrainData.new('not_a_hash') } 
    
    # Bad key in hash:
    assert_raises(RuntimeError) { RubyFann::TrainData.new({:not_a_real_value=>1}) } 
    
    # Bad value in hash:
    assert_raises(RuntimeError) { RubyFann::TrainData.new({:inputs=>1, :desired_outputs=>2}) }
    
    # Inconsistent inputs
    assert_raises(RuntimeError) { 
      RubyFann::TrainData.new(
        { 
          :inputs=>[[0.2, 0.3, 0.4], [0.8]], 
          :desired_outputs=>[[3.14], [4.33]]
        }
      ) 
    } 
    
    # Inconsistent outputs
    assert_raises(RuntimeError) { 
      RubyFann::TrainData.new(
        { 
          :inputs=>[[0.2, 0.3, 0.4], [0.5, 0.8, 0.7]], 
          :desired_outputs=>[[3.14], [0.4, 0.5]]
        }
      ) 
    }
    
    # No errors:
    assert_nothing_raised(){ 
      RubyFann::TrainData.new(
        { 
          :inputs=>[[0.2, 0.3, 0.4], [0.8, 0.9, 0.7]], 
          :desired_outputs=>[[3.14], [6.33]]
        }
      ) 
    }
    
  end

private
  # Set & get fann attribute & verify:
  def verify_fann_attribute(attr, val)
    fann = RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[3, 4, 3, 4], :num_outputs=>1)
    fann.send("set_#{attr}", val)    
    assert_equal(val, fann.send("get_#{attr}"))    
  end
  
end

