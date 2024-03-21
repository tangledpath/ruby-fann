# -*- encoding: utf-8 -*-
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'ruby_fann/version'

Gem::Specification.new do |gem|
  gem.name              = "ruby-fann"
  gem.version           = RubyFann::VERSION::STRING
  gem.authors           = ["tangledpath"]
  gem.email             = ["steven.miers@gmail.com"]
  gem.license           = 'MIT'
  gem.description       = %q{Bindings to use FANN from within ruby/rails environment}
  gem.summary           = %q{Bindings to use FANN from within ruby/rails environment.  Fann is a is a free open source neural network library, which implements multilayer artificial neural networks with support for both fully connected and sparsely connected networks.  It is easy to use, versatile, well documented, and fast.  RubyFann makes working with neural networks a breeze using ruby, with the added benefit that most of the heavy lifting is done natively.}
  gem.homepage          = "http://github.com/tangledpath/ruby-fann"
  gem.rubyforge_project = 'ruby-fann'
  gem.extra_rdoc_files  = ['README.md', 'ext/ruby_fann/ruby_fann.c']
  gem.files             =  Dir.glob('lib/**/*.rb') + Dir.glob('ext/**/*.{c,h,rb}')
  gem.extensions        = ['ext/ruby_fann/extconf.rb']
  gem.executables       = gem.files.grep(%r{^bin/}).map{ |f| File.basename(f) }
  gem.test_files        = gem.files.grep(%r{^(test|spec|features)/})
  gem.require_paths     = ["lib", "ext"]
  gem.metadata          = {
    "documentation_uri" => "https://tangledpath.github.io/ruby-fann/"
  }
end
