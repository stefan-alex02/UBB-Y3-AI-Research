package ro.ubb.ai.javaserver.config;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import ro.ubb.ai.javaserver.security.AuthEntryPointJwt;
import ro.ubb.ai.javaserver.security.AuthTokenFilter;
import ro.ubb.ai.javaserver.security.UserDetailsServiceImpl;
// No MvcRequestMatcher import needed here for the new approach

@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
@RequiredArgsConstructor
public class SecurityConfig {

    private final UserDetailsServiceImpl userDetailsService; // Kept if still used for UserDetails loading
    private final AuthEntryPointJwt unauthorizedHandler;
    private final AuthTokenFilter authTokenFilter;

    // Make expectedInternalApiKey accessible for custom access checks if needed elsewhere
    // (e.g., if a custom filter checks the header for /api/internal/**)
    @Getter
    @Value("${python.api.internal-key}")
    private String expectedInternalApiKey;
    // private static final String INTERNAL_API_KEY_HEADER = "X-Internal-API-Key"; // Not directly used in this config variant


    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.csrf(AbstractHttpConfigurer::disable)
                .exceptionHandling(exception -> exception.authenticationEntryPoint(unauthorizedHandler))
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        // Allow OPTIONS requests for preflight checks globally or for specific API paths
                        .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll() // Broadest, good for development
                        // Or more specific:
                        // .requestMatchers(HttpMethod.OPTIONS, "/api/**").permitAll()

                        .requestMatchers("/api/auth/**").permitAll()
                        .requestMatchers("/api/internal/**").permitAll() // Still needs its own security (e.g., API key filter)
                        .requestMatchers("/api/**").authenticated() // All other /api paths require authentication
                        .anyRequest().permitAll() // Or .denyAll() / .authenticated() as per your default
                );

        http.addFilterBefore(authTokenFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
