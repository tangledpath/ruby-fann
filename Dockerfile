FROM ruby:3.2

# Copy function code
RUN mkdir /ruby-fann
COPY . /ruby-fann
# RUN gem install ruby-fann
# RUN bundle install
