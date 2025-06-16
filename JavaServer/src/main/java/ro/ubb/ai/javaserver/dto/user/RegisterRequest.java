package ro.ubb.ai.javaserver.dto.user;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;
import ro.ubb.ai.javaserver.enums.Role;

@Data
public class RegisterRequest {
    @NotBlank
    @Size(min = 2, max = 100) // TODO change min back to 3
    private String username;

    @NotBlank
    @Size(min = 2, max = 40) // TODO change min back to 6
    private String password;

    @Size(max = 255)
    private String name;

    private Role role = Role.NORMAL;

    private String meteorologistPasscode;
}
