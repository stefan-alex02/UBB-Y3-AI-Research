package ro.ubb.ai.javaserver.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig {

    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/api/**") // Apply to all paths under /api
                        .allowedOrigins("http://localhost:3000") // Your React app's origin
                        .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS") // Allowed HTTP methods
                        .allowedHeaders("*") // Allow all headers (you can be more specific)
                        .allowCredentials(true) // If you need to send cookies or use Authorization headers with credentials
                        .maxAge(3600); // Cache preflight response for 1 hour
            }
        };
    }
}