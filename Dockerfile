FROM ruby:3.0

# Copy function code
# COPY test/Gemfile* /
RUN gem install ruby-fann
# RUN bundle install
