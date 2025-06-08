package ro.ubb.ai.javaserver.websocket;

import ro.ubb.ai.javaserver.dto.experiment.ExperimentDTO; // Your existing DTO
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Component
@Slf4j
@RequiredArgsConstructor
public class ExperimentStatusWebSocketHandler extends TextWebSocketHandler {

    // Thread-safe list to store active sessions
    private final List<WebSocketSession> sessions = new CopyOnWriteArrayList<>();
    private final ObjectMapper objectMapper; // For serializing ExperimentDTO to JSON

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        sessions.add(session);
        log.info("WebSocket connection established: Session ID {}, Remote: {}", session.getId(), session.getRemoteAddress());
        // Optionally, send a welcome message or initial data if needed
        // session.sendMessage(new TextMessage("Connected to experiment status updates."));
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        // Handle incoming messages from clients if any (e.g., subscription requests)
        // For this use case, we might not need clients to send messages, only receive.
        String payload = message.getPayload();
        log.info("Received WebSocket message from {}: {}", session.getId(), payload);
        // Example: session.sendMessage(new TextMessage("Message received: " + payload));
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        sessions.remove(session);
        log.info("WebSocket connection closed: Session ID {}, Status: {}", session.getId(), status);
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        log.error("WebSocket transport error for session {}: {}", session.getId(), exception.getMessage());
        if (session.isOpen()) {
            session.close(CloseStatus.SERVER_ERROR);
        }
        sessions.remove(session);
    }

    /**
     * Broadcasts an updated ExperimentDTO to all connected WebSocket clients.
     * This method will be called by the ExperimentService when an experiment's status changes.
     *
     * @param experimentDTO The updated experiment data.
     */
    public void broadcastExperimentUpdate(ExperimentDTO experimentDTO) {
        if (experimentDTO == null) {
            log.warn("Attempted to broadcast a null experimentDTO.");
            return;
        }

        try {
            String messagePayload = objectMapper.writeValueAsString(experimentDTO);
            TextMessage message = new TextMessage(messagePayload);

            // Iterate over a copy of the sessions list to avoid ConcurrentModificationException
            // if a session closes while broadcasting.
            for (WebSocketSession session : new ArrayList<>(sessions)) {
                try {
                    if (session.isOpen()) {
                        session.sendMessage(message);
                        log.debug("Broadcasted experiment update for ID {} to session {}", experimentDTO.getExperimentRunId(), session.getId());
                    }
                } catch (IOException e) {
                    log.error("Error sending WebSocket message to session {}: {}", session.getId(), e.getMessage());
                    // Optionally remove session if sending fails persistently
                }
            }
            log.info("Broadcasted update for experiment: {}", experimentDTO.getExperimentRunId());
        } catch (JsonProcessingException e) {
            log.error("Error serializing ExperimentDTO to JSON for WebSocket broadcast: {}", e.getMessage(), e);
        }
    }
}
