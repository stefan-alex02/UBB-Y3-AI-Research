package ro.ubb.ai.javaserver.config;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import ro.ubb.ai.javaserver.websocket.ExperimentStatusWebSocketHandler;

@Configuration
@EnableWebSocket
@RequiredArgsConstructor
public class WebSocketConfig implements WebSocketConfigurer {

    @Value("${client.origin}")
    private String allowedOrigin;

    private final ExperimentStatusWebSocketHandler experimentStatusWebSocketHandler;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(experimentStatusWebSocketHandler, "/ws/experiment-status")
                .setAllowedOrigins(allowedOrigin);
    }
}
