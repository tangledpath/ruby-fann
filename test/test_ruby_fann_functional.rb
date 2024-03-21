require 'test/unit'

class RubyFannFunctionalTest < Test::Unit::TestCase
  XOR_INPUT_DATA =  [[-1, -1], [-1, 1], [1, -1], [1, 1]]
  XOR_OUTPUT_DATA = [[-1.0],   [1.0],   [1.0],   [-1.0]]

  def test_training_xor
    nn = RubyFann::Standard.new(:num_inputs=>2, :hidden_neurons=>[3], :num_outputs=>1)


    training = RubyFann::TrainData.new(:inputs=>XOR_INPUT_DATA, :desired_outputs=>XOR_OUTPUT_DATA)

    nn.set_activation_steepness_hidden(1.0)
  	nn.set_activation_steepness_output(1.0)


  	nn.set_activation_function_hidden(:sigmoid_symmetric)
  	nn.set_activation_function_output(:sigmoid_symmetric)

#    nn.set_train_error_function(:tanh)
    bit_fail = 0.001
  	nn.set_train_stop_function(:bit)
  	nn.set_bit_fail_limit(bit_fail)

  	nn.init_weights(training)

  	puts("Training network.\n")


  	nn.train_on_data(training, 10000, 1000, 0.01)
  	printf("Saving network.\n")
  	nn.save("xor_float.net")
  	training.save("xor.train")
  	assert(File.exist?('xor_float.net'))

    verify_training_data(nn, training, XOR_INPUT_DATA, XOR_OUTPUT_DATA)

  end

  def test_training_xor_network_loading
    nn = RubyFann::Standard.new(:filename=>'xor_float.net')
    training = RubyFann::TrainData.new(:filename=>'xor.train')
    verify_training_data(nn, training, XOR_INPUT_DATA, XOR_OUTPUT_DATA)
  end

  def test_cascade_training_xor
    nn = RubyFann::Shortcut.new(:num_inputs=>2, :num_outputs=>1)

    training = RubyFann::TrainData.new(:inputs=>XOR_INPUT_DATA, :desired_outputs=>XOR_OUTPUT_DATA)

  	puts("Training network.\n")

    nn.set_activation_function_output(:sigmoid_symmetric)
    #nn.set_cascade_activation_functions([:sigmoid, :sigmoid_symmetric])
    nn.set_train_error_function(:linear)
  	nn.cascadetrain_on_data(training, 100, 1, 0.0);
  	nn.save("xor_cascade.net")

    verify_training_data(nn, training, XOR_INPUT_DATA, XOR_OUTPUT_DATA, expected_error=0.00000001)
  end

  def test_cascade_training_xor_network_loading
    nn = RubyFann::Standard.new(:filename=>'xor_cascade.net')
    training = RubyFann::TrainData.new(:filename=>'xor.train')
    verify_training_data(nn, training, XOR_INPUT_DATA, XOR_OUTPUT_DATA)
  end

private
  def verify_training_data(nn, training, input_data, output_data, expected_error=0.002)
    mse = nn.test_data(training)
  	puts("Tested network. %f\n" % mse)

    calc_out = []
  	0.upto(input_data.length-1) do |i|
  	  c = nn.run(input_data[i])
  		calc_out << c
      puts("XOR test (%f,%f) -> %f, should be %f, difference=%f\n" %
           [input_data[i][0], input_data[i][1], c[0], output_data[i][0],
           (c[0] - output_data[i][0]).abs])
  	end

  	0.upto(input_data.length-1) {|i| assert_in_delta(output_data[i][0], calc_out[i][0], expected_error)}
  end
end