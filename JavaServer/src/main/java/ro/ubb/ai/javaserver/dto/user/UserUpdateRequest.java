package ro.ubb.ai.javaserver.dto.user;

import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class UserUpdateRequest {
    @Size(max = 255)
    private String name;

    @Size(min = 6, max = 40)
    private String newPassword;
}
