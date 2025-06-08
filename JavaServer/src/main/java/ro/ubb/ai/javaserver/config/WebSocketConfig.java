package ro.ubb.ai.javaserver.config;

import org.springframework.beans.factory.annotation.Value;
import ro.ubb.ai.javaserver.websocket.ExperimentStatusWebSocketHandler; // We will create this
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
@RequiredArgsConstructor
public class WebSocketConfig implements WebSocketConfigurer {

    @Value("${client.origin}") // Inject the allowed origin from application properties
    private String allowedOrigin; // This can be used if you need to reference it elsewhere

    private final ExperimentStatusWebSocketHandler experimentStatusWebSocketHandler; // Inject your handler

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(experimentStatusWebSocketHandler, "/ws/experiment-status")
                // Allow connections from your React frontend origin
                .setAllowedOrigins(allowedOrigin); // Use a property for the origin
        // For production, be more specific or use a list of allowed origins
        // You can also add .withSockJS() here if you want SockJS fallback for older browsers,
        // but modern browsers usually support WebSockets directly.
    }
}
