package ro.ubb.ai.javaserver.dto.user;

import ro.ubb.ai.javaserver.enums.Role;
import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.Email; // If username is email

@Data
public class MessageResponse { // Generic message response
    private String message;
    public MessageResponse(String message) { this.message = message; }
}
