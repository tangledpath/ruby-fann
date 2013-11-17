# RubyFann
*Artifical Intelligence in Ruby*

[![Gem Version](https://badge.fury.io/rb/ruby-fann.png)](http://badge.fury.io/rb/ruby-fann)

RubyFann, or "ruby-fann" is a ruby gem that binds to FANN (Fast Artificial Neural Network) from within a ruby/rails environment.  FANN is a is a free open source neural network library, which implements multilayer artificial neural networks with support for both fully connected and sparsely connected networks.  It is easy to use, versatile, well documented, and fast.  RubyFann makes working with neural networks a breeze using ruby, with the added benefit that most of the heavy lifting is done natively.

A talk given by our friend Ethan from Big-Oh Studios at Lone Star Ruby 2013: http://confreaks.com/videos/2609-lonestarruby2013-neural-networks-with-rubyfann

## Installation

Add this line to your application's Gemfile:

    gem 'ruby-fann'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install ruby-fann

## Usage

First, Go here & read about FANN. You don't need to install it before using the gem, but understanding FANN will help you understand what you can do with the ruby-fann gem:
http://leenissen.dk/fann/

ruby-fann RDocs:
http://ruby-fann.rubyforge.org/

### Example training & subsequent execution:
  
```ruby
  require 'ruby-fann'
  train = RubyFann::TrainData.new(:inputs=>[[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]], :desired_outputs=>[[0.7], [0.8]])
  fann = RubyFann::Standard.new(:num_inputs=>3, :hidden_neurons=>[2, 8, 4, 3, 4], :num_outputs=>1)
  fann.train_on_data(train, 1000, 10, 0.1) # 1000 max_epochs, 10 errors between reports and 0.1 desired MSE (mean-squared-error)
  outputs = fann.run([0.3, 0.2, 0.4])    
```

### Save training data to file and use it later (continued from above)

```ruby
  train.save('verify.train')
  train = RubyFann::TrainData.new(:filename=>'verify.train')
  # Train again with 10000 max_epochs, 20 errors between reports and 0.01 desired MSE (mean-squared-error)
  # This will take longer:
  fann.train_on_data(train, 10000, 20, 0.01) 
```

### Save trained network to file and use it later (continued from above)

```ruby
  fann.save('foo.net')
  saved_nn = RubyFann::Standard.new(:filename=>"foo.net")
  saved_nn.run([0.3, 0.2, 0.4])  
```
  
### Custom training using a callback method

This callback function can be called during training when using train_on_data, train_on_file or cascadetrain_on_data.

It is very useful for doing custom things during training.  It is recommended to use this function when implementing custom training procedures, or when visualizing the training in a GUI etc.  The args which the callback function takes is the parameters given to the train_on_data, plus an epochs parameter which tells how many epochs the training have taken so far.

The callback method should return an integer, if the callback function returns -1, the training will terminate.

The callback (training_callback) will be automatically called if it is implemented on your subclass as follows:

```ruby
class MyFann < RubyFann::Standard
  def training_callback(args)
    puts "ARGS: #{args.inspect}"
    0  
  end
end
```
### A sample project using RubyFann to play tic-tac-toe
https://github.com/bigohstudios/tictactoe

## Contributors
1. Steven Miers
2. Ole Krüger
3. dignati
4. Michal Pokorny
5. Scott Li (locksley)

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
