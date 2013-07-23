require "rubygems" 
require "graphviz"

module RubyFann
  # Generates directed graph from a RubyFann neural network.
  # Requires the GraphViz gem 0.8.0 (or higher) to be installed, 
  # as well as graphviz proper 2.14.1 (or higher).
  class Neurotica # :nodoc:
    attr_accessor :connector_colors
    attr_accessor :input_layer_color
    attr_accessor :hidden_layer_colors
    attr_accessor :output_layer_color
    
    # Initialize neurotica grapher with the following args:
    #   :connector_colors    - array of graphviz-friendly color names/numbers
    #   :input_layer_color   - graphviz-friendly color name/number
    #   :hidden_layer_colors - array of graphviz-friendly color names/numbers
    #   :output_layer_color  - graphviz-friendly color name/number
    def initialize(args={})      
      @connector_colors = args[:connector_colors]
      @input_layer_color = args[:input_layer_color]
      @hidden_layer_colors = args[:hidden_layer_colors]
      @output_layer_color = args[:output_layer_color]
      @connector_colors ||= ['red', 'blue', 'yellow', 'green', 'orange', 'black', 'pink', 'gold', 'lightblue', 'firebrick4', 'purple']   
      @input_layer_color ||= 'green'
      @hidden_layer_colors ||= ['bisque2', 'yellow', 'blue', 'orange', 'khaki3']   
      @output_layer_color ||= 'purple'
    end
    
    # Generate output graph with given neural network to the given output path (PNG)
    #   If args[:three_dimensional] is set, then a 3d VRML graph will be generated (experimental)
    def graph(neural_net, output_path, args={})
      if (args[:three_dimensional])
        graph_viz = GraphViz::new( "G", :dim=>'3') # , :size=>"17,11"     
        shape="point"
      else
        graph_viz = GraphViz::new( "G", :dim=>'2') # , :size=>"17,11"     
        shape="egg"
      end
      
      neurons = neural_net.get_neurons()
      graph_node_hash = {}
      max_layer = neurons.max {|a,b| a[:layer] <=> b[:layer] }[:layer]


      # Add nodes:
      neurons.each do |neuron|
        fillcolor = "transparent"  # : "khaki3"
        layer = neuron[:layer]
        fillcolor = case layer
          when 0
            @input_layer_color
          when max_layer
            @output_layer_color
          else
            @hidden_layer_colors[(layer-1) % @hidden_layer_colors.length]
        end
        
        #puts "adding neuron with #{neuron[:value]}"
        node_id = neuron.object_id.to_s
#        label = (layer==0) ? ("%d-%0.3f-%0.3f" % [neuron[:layer], neuron[:value], neuron[:sum]]) : ("%d-%0.3f-%0.3f" % [neuron[:layer], neuron[:value], neuron[:sum]])       
        label = (layer==0 || layer==max_layer) ? ("%0.3f" % neuron[:value]) : ("%0.3f" % rand) #neuron[:sum]) 
        graph_node_hash[node_id] = graph_viz.add_node(
          node_id,
          :label=>label,
          :style=>"filled",
          :fillcolor=>fillcolor,
          #:color=>fillcolor,
          :shape=>shape,
          :z=>"#{rand(100)}", # TODO
          # :z=>"0", # TODO
          # :width=>"1",
          # :height=>"1",
          :fontname=>"Verdana"
        )        
      end
            
      previous_neurons = nil
      layer_neurons = nil
      0.upto(max_layer) do |layer|
        previous_neurons = layer_neurons
        layer_neurons = neurons.find_all{|n| n[:layer]==layer}
        
        if previous_neurons
          previous_neurons.each do |pn|
            node_id = pn.object_id.to_s
            
            layer_neurons.each do |n|
              dest_id = n.object_id.to_s
              graph_viz.add_edge(
                graph_node_hash[node_id], 
                graph_node_hash[dest_id], 
                :weight=>"10",
                :color=>"#{connector_colors[layer % connector_colors.length]}"
              )                      
            end
          end
        end
        
      end      
      
      if (args[:three_dimensional])
        graph_viz.output(:vrml=>output_path)
      else
        graph_viz.output(:png=>output_path)
      end
      
      
    end
  end
end
