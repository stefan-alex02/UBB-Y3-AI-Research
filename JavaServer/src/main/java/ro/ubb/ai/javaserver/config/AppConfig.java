package ro.ubb.ai.javaserver.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.ClientHttpRequestInterceptor;
import org.springframework.http.converter.ByteArrayHttpMessageConverter;
import org.springframework.http.converter.ResourceHttpMessageConverter;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.http.converter.support.AllEncompassingFormHttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpRequest;
import org.springframework.http.client.BufferingClientHttpRequestFactory;
import org.springframework.http.client.ClientHttpRequestExecution;
import org.springframework.http.client.ClientHttpRequestInterceptor;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.http.client.SimpleClientHttpRequestFactory; // Or your preferred factory
import org.springframework.util.StreamUtils;

import java.io.IOException;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

@Configuration
public class AppConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder, ObjectMapper objectMapper) {
        // Wrap the default factory or your chosen factory with BufferingClientHttpRequestFactory
        SimpleClientHttpRequestFactory simpleFactory = new SimpleClientHttpRequestFactory();
//        simpleFactory.setBufferRequestBody(false); // Important for BufferingClientHttpRequestFactory to work correctly with streaming
        BufferingClientHttpRequestFactory bufferingFactory = new BufferingClientHttpRequestFactory(simpleFactory);

        RestTemplate restTemplate = builder
                .requestFactory(() -> bufferingFactory) // Use the buffering factory
                .build();

        // Add the interceptor
        List<ClientHttpRequestInterceptor> interceptors = restTemplate.getInterceptors();
        if (interceptors == null || interceptors.isEmpty()) {
            interceptors = new ArrayList<>();
        }
        interceptors.add(new LoggingInterceptor());
        restTemplate.setInterceptors(interceptors);

        // Set message converters
        List<org.springframework.http.converter.HttpMessageConverter<?>> messageConverters = new ArrayList<>();
        messageConverters.add(new ByteArrayHttpMessageConverter());
        messageConverters.add(new StringHttpMessageConverter(StandardCharsets.UTF_8));
        messageConverters.add(new ResourceHttpMessageConverter(false));
        messageConverters.add(new AllEncompassingFormHttpMessageConverter());
        messageConverters.add(new MappingJackson2HttpMessageConverter(objectMapper));
        restTemplate.setMessageConverters(messageConverters);

        return restTemplate;
    }

    @Bean
    public ObjectMapper objectMapper() {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        // Configure Jackson to convert Java camelCase to JSON snake_case
        objectMapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE); // <--- ADD THIS LINE
        // objectMapper.configure(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false); // Already good
        return objectMapper;
    }
}

@Slf4j // Add lombok @Slf4j to AppConfig if not already there
class LoggingInterceptor implements ClientHttpRequestInterceptor {
    @Override
    public ClientHttpResponse intercept(HttpRequest request, byte[] body, ClientHttpRequestExecution execution) throws IOException {
        logRequest(request, body);
        ClientHttpResponse response = execution.execute(request, body);
        logResponse(response);
        return response;
    }

    private void logRequest(HttpRequest request, byte[] body) throws IOException {
        log.debug("===========================request begin================================================");
        log.debug("URI         : {}", request.getURI());
        log.debug("Method      : {}", request.getMethod());
        log.debug("Headers     : {}", request.getHeaders());
        log.debug("Request body: {}", new String(body, StandardCharsets.UTF_8));
        log.debug("==========================request end================================================");
    }

    private void logResponse(ClientHttpResponse response) throws IOException {
        // Buffer the response body so it can be read multiple times (e.g., by logger and by RestTemplate)
        // This requires BufferingClientHttpRequestFactory
        log.debug("============================response begin==========================================");
        log.debug("Status code  : {}", response.getStatusCode());
        log.debug("Status text  : {}", response.getStatusText());
        log.debug("Headers      : {}", response.getHeaders());
        // Be careful logging response body if it's large or binary
        // String responseBody = StreamUtils.copyToString(response.getBody(), StandardCharsets.UTF_8);
        // log.debug("Response body: {}", responseBody);
        log.debug("=======================response end=================================================");
    }
}