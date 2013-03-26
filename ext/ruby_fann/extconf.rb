require 'mkmf'
# lib = dir_config('fann', '/usr/local')
#if !have_library("doublefann", "fann_create_standard")
#  puts "FANN must be installed and available in /usr/local or passed in with --with-fann-dir.  Windows users should use ruby compiled in Cygwin or an equivalent, such as MingW.  Ruby installed with the OneClickInstaller is not sufficient."
#  exit 1
#end
#find_library("doublefann", "fann_create_standard", "/usr/local/lib")

#dir_config('fann', '.')
$objs=["ruby_fann.o", "doublefann.o"]
have_header("doublefann.h")
create_makefile("ruby_fann")